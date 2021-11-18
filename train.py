import os
import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
import argparse
import random
import argparse

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import wandb
from dataset import *
from model import *

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_config():
    parser = argparse.ArgumentParser()


    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--save_dir', type=str, default = './best_model/fold', 
                        help='model save dir path (default : ./best_model/fold)')
    parser.add_argument('--wandb_path', type= str, default= 'test',
                        help='wandb graph, save_dir basic path (default: sm_kr_punc_lstm') 
    parser.add_argument('--train_path', type= str, default= '/opt/ml/test_code/data/train.csv',
                        help='train csv path (default: /opt/ml/dataset/train/train.csv')
    parser.add_argument('--tokenize_option', type=str, default='SUB',
                        help='token option ex) SUB, PUN')    
    parser.add_argument('--fold', type=int, default=5,
                        help='fold (default: 5)')
    parser.add_argument('--model', type=str, default='klue/roberta-large',
                        help='model type (default: klue/roberta-large)')
    parser.add_argument('--loss', type=str, default= 'LB',
                        help='LB: LabelSmoothing, CE: CrossEntropy')


    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--batch', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--gradient_accum', type=int, default=2,
                        help='gradient accumulation (default: 2)')
    parser.add_argument('--batch_valid', type=int, default=32,
                        help='input batch size for validing (default: 32)')
    parser.add_argument('--warmup', type=int, default=0.1,
                        help='warmup_ratio (default: 0.1)')
    parser.add_argument('--eval_steps', type=int, default=20,
                        help='eval_steps (default: 250)')
    parser.add_argument('--save_steps', type=int, default=20,
                        help='save_steps (default: 250)')
    parser.add_argument('--logging_steps', type=int,
                        default=10, help='logging_steps (default: 50)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='weight_decay (default: 0.01)')
    parser.add_argument('--metric_for_best_model', type=str, default='accuracy',
                        help='metric_for_best_model (default: micro f1 score')
    
    args= parser.parse_args()

    return args

class Custom_Trainer(Trainer):
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name
    
    def compute_loss(self, model, inputs, return_outputs= False):
        labels= inputs.pop('labels')
        outputs= model(**inputs)
        device= torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')
        
        if self.args.past_index >=0:
            self._past= outputs[self.args.past_index]

        if self.loss_name== 'CrossEntropyLoss':
            custom_loss= torch.nn.CrossEntropyLoss().to(device)
            loss= custom_loss(outputs['logits'], labels)
        
        elif self.loss_name== 'LabelSmoothLoss' and self.label_smoother is not None:
            loss= self.label_smoother(outputs, labels)
            loss= loss.to(device)
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    acc = accuracy_score(labels, preds) 

    return {
        'accuracy': acc * 100,
    }

def train(args):
    
    seed_everything(args.seed)
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer= AutoTokenizer.from_pretrained(args.model)
    preprocess= Preprocess(args.train_path, args.tokenize_option)

    all_dataset= preprocess.data
    all_label= all_dataset['label'].values

    kfold= StratifiedKFold(n_splits= 5, shuffle= True, random_state= 42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_dataset, all_label)):
        run= wandb.init(project= 'dataset', entity= 'quarter100', name= f'KFOLD_{fold}_{args.wandb_path}')
        print(f'fold: {fold} start!')
        train_dataset= all_dataset.iloc[train_idx]
        val_dataset= all_dataset.iloc[val_idx]

        train_label= train_dataset['label'].values
        val_label= val_dataset['label'].values

        tokenized_train= preprocess.tokenized_dataset(train_dataset, tokenizer)
        tokenized_val= preprocess.tokenized_dataset(val_dataset, tokenizer)

        trainset= Dataset(tokenized_train, train_label)
        valset= Dataset(tokenized_val, val_label)

        model= Model(args.model)
        # model.model.resize_token_embeddings(tokenizer.vocab_size + token_size)
        model.to(device)

        save_dir= f'./result/KFOLD_{fold}_{args.wandb_path}'

        training_args= TrainingArguments(
            output_dir= save_dir,
            save_total_limit= 1,
            # gradient_accumulation_steps= args.gradient_accum,
            save_steps=args.save_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch_valid,
            label_smoothing_factor=0.1,
            warmup_ratio= args.warmup,
            weight_decay=args.weight_decay,
            logging_dir='./logs',
            logging_steps=args.logging_steps,
            metric_for_best_model= args.metric_for_best_model,
            evaluation_strategy= 'steps',
            group_by_length= True,
            eval_steps= args.eval_steps,
            load_best_model_at_end=True
        )

        if args.loss== 'LB':
            trainer= Trainer(
                model= model,
                args= training_args,
                train_dataset= trainset,
                eval_dataset= valset,
                compute_metrics= compute_metrics,
                callbacks= [EarlyStoppingCallback(early_stopping_patience= 3)]
            )

        elif args.loss== 'CE':
            trainer= Custom_Trainer(
                model=model,                         # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=trainset,         # training dataset
                eval_dataset=valset,             # evaluation dataset
                compute_metrics=compute_metrics,         # define metrics function
                callbacks = [EarlyStoppingCallback(early_stopping_patience= 3)],
                loss_name = 'CrossEntropyLoss'
            )

        trainer.train()
        if not os.path.exists(f'{args.save_dir}_{fold}'):
            os.makedirs(f'{args.save_dir}_{fold}')
        torch.save(model.state_dict(), os.path.join(f'{args.save_dir}_{fold}', 'pytorch_model.bin'))
        run.finish()
        print(f'fold{fold} fin!')
        break;


if __name__ == '__main__':

    args= get_config()
    train(args)