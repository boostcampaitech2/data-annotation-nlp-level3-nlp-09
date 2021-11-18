from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel

from dataset import *
from model import *

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

def get_test_config():
    parser= argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--kfold', type=int, default=5,
                        help='kfold (default: 5)')                   
    parser.add_argument('--model_path', type=str, default = './best_model/fold', 
                        help='model load dir path (default : ./best_model/fold)')
    parser.add_argument('--save_dir', type=str, 
                        default='./prediction',
                        help='submission save path')     
    parser.add_argument('--batch', type=int, default=64,
                        help='input batch size for test (default: 6432)')
    parser.add_argument('--add_token', type=int, default=6,
                        help='add token count (default: 6)')
    parser.add_argument('--tokenize_option', type=str, default='PUN',
                        help='token option ex) SUB, PUN')    
    parser.add_argument('--test_path', type=str, 
                        default='/opt/ml/test_code/data/test.csv',
                        help='test csv path') 

    args= parser.parse_args()

    return args

def inference(model, tokenized_data, device, args):
    dataloader= DataLoader(tokenized_data, batch_size= args.batch, shuffle= False)
    model.eval()
    output_pred, output_prob= [], []

    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs= model(
                input_ids= data['input_ids'].to(device),
                attention_mask= data['attention_mask'].to(device)
            )
        logits= outputs['logits']
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits= logits.detach().cpu().numpy()
        # prob= logits

        result= np.argmax(logits, axis= -1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis= 0).tolist()



def load_test_dataset(dataset_dir, tokenizer, args):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """

    preprocess= Preprocess(dataset_dir, args.tokenize_option)
    
    test_dataset = preprocess.load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))
    tokenized_test= preprocess.tokenized_dataset(test_dataset, tokenizer)
    
    # print(test_dataset)
    return test_dataset['id'], tokenized_test, test_label

def to_nparray(s) :
    return np.array(list(map(float, s[1:-1].split(','))))

def main_inference(args):
    print('main inference start')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer= AutoTokenizer.from_pretrained(args.model)

    df_list= []
    i= 0
    print(f'KFOLD : {i} inference start !')
    model= Model(args.model)

    best_state_dict= torch.load(os.path.join(f'{args.model_path}_{i}', 'pytorch_model.bin'))
    model.load_state_dict(best_state_dict)
    model.to(device)
    
    test_id, test_dataset, test_label= load_test_dataset(args.test_path, tokenizer, args)
    # print('test', test_id)
    # print(test_dataset)
    # print(test_label)
    testset= Dataset(test_dataset, test_label)

    pred_answer, output_prob= inference(model, testset, device, args)
    print(len(test_id), len(pred_answer), len(output_prob))

    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    output.to_csv(os.path.join(args.save_dir, f'submission{i}.csv'), index= False)
    
    print(f'KFOLD : {i} inference fin !')

    print('FIN')


if __name__ == '__main__':
    args= get_test_config()
    main_inference(args)
