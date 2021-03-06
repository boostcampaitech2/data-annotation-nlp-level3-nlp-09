import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer
import pickle

class Preprocess:
    def __init__(self, CSV_PATH, version):
        self.version= version
        self.data= self.load_data(CSV_PATH)

    def load_data(self, path):
        
        data= pd.read_csv(path)
        
        sub_entity, sub_type= [], []
        obj_entity, obj_type= [], []
        sub_idx, obj_idx= [], []
        sentence= []

        """preprocess"""
        for i, [x, y, z] in enumerate(zip(data['subject_entity'], data['object_entity'], data['sentence'])):
            sub_typ= x[1:-1].split(':')[-1].split('\'')[-2]
            obj_typ= y[1:-1].split(':')[-1].split('\'')[-2]
            # print(x, y)
            for idx_i in range(len(x)):
                if x[idx_i: idx_i+ 5]== 'start':
                    sub_start= int(x[idx_i+7:].split(',')[0].strip())
                if x[idx_i: idx_i+4]== 'text':
                    sub_text= x[idx_i +6:].split(',')[0].strip()[1:-1]
                    sub_end= sub_start+ len(sub_text)
                
                if y[idx_i: idx_i+ 5]== 'start':
                    obj_start= int(y[idx_i+7:].split(',')[0].strip())
                if y[idx_i: idx_i+4]== 'text':
                    obj_text= y[idx_i +6:].split(',')[0].strip()[1:-1]
                    obj_end= obj_start+ len(obj_text)
            
            sub_i= [sub_start, sub_end]
            obj_i= [obj_start, obj_end]

            # print(z[sub_i[0]: sub_i[1]])
            # print(z[obj_i[0]: obj_i[1]])
            sub_entity.append(z[sub_i[0]: sub_i[1]])
            obj_entity.append(z[obj_i[0]: obj_i[1]])
            sub_type.append(sub_typ); sub_idx.append(sub_i)
            obj_type.append(obj_typ); obj_idx.append(obj_i)
            
            # """tokenize version"""
            # if self.version== 'SUB':
            #     if sub_i[0] < obj_i[0]:
            #         z= z[:sub_i[0]] + '[SUB]'+ z[sub_i[0]: sub_i[1]+1] + '[/SUB]' + z[sub_i[1]+1:]
            #         z= z[:obj_i[0]+11] + '[OBJ]'+ z[obj_i[0]+11: obj_i[1]+12]+ '[/OBJ]'+ z[obj_i[1]+12:]
            #     else:
            #         z= z[:obj_i[0]] + '[OBJ]'+ z[obj_i[0]: obj_i[1]+1]+ '[/OBJ]'+ z[obj_i[1]+1:]
            #         z= z[:sub_i[0]+11] + '[SUB]'+ z[sub_i[0]+11: sub_i[1]+12] + '[/SUB]' + z[sub_i[1]+12:]

            # elif self.version== 'PUN':
            #     if sub_i[0] < obj_i[0]:
            #         z= z[:sub_i[0]] + '@*'+sub_typ+'*'+ z[sub_i[0]: sub_i[1]+1] + '@' + z[sub_i[1]+1:]
            #         z= z[:obj_i[0]+7] + '#^'+ obj_typ +'^'+ z[obj_i[0]+7: obj_i[1]+8]+ '#'+ z[obj_i[1]+8:]
            #     else:
            #         z= z[:obj_i[0]] + '#^'+ obj_typ +'^'+ z[obj_i[0]: obj_i[1]+1]+ '#' + z[obj_i[1]+1:]
            #         z= z[:sub_i[0]+7] + '@*'+sub_typ+'*' + z[sub_i[0]+7: sub_i[1]+8] + '@' + z[sub_i[1]+8:]

            sentence.append(z)

        df= pd.DataFrame({'id': data['idx'], 'sentence' : sentence, 'subject_entity': sub_entity, 'object_entity': obj_entity,
                                'subject_type': sub_type, 'object_type': obj_type, 'label': data['class'],
                                'subject_idx': sub_idx, 'object_idx': obj_idx})
        print(df)
        
        return df
    
    def tokenized_dataset(self, data, tokenizer):

        concat_entity = []
        for sub_ent, obj_ent, sub_typ, obj_typ in zip(data['subject_entity'], data['object_entity'], data['subject_type'], data['object_type']):
            temp =  '@*'+ sub_typ + '*' + sub_ent + '@??? #^' + obj_typ + '^' + obj_ent + '#??? ??????'
            #temp =  e01 + '???' + e02 + '??? ??????'
            concat_entity.append(temp)

        tokenized_sentence= tokenizer(
            concat_entity,
            list(data['sentence']), # list??? string type?????? ???????????? ??? !
            return_tensors= "pt", # pytorch type
            padding= True, # ????????? ????????? ????????? padding
            truncation= True, # ?????? ?????????
            max_length= 256, # ?????? ?????? ??????...
            add_special_tokens= True, # special token ??????
            return_token_type_ids= False # roberta??? ??????.. token_type_ids??? ???????????? ! 
        )    

        return tokenized_sentence
    


"""Train, Test Dataset"""
class Dataset:
    def __init__(self, data, labels): # data : dict, label : list??????..
        self.data= data
        self.labels= labels
    
    def __getitem__(self, idx):
        # print(self.data)
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item
    
    def __len__(self):
        return len(self.labels)

