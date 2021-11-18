import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# def make_drop_data(data):
#     df= data.dropna(axis= 0)
#     # print(df.isnull().sum())
#     df.to_csv('/opt/ml/test_code/data/q100_drop.csv', index= False)

# if __name__ == '__main__':
#     # org data 1039
#     data= pd.read_csv('./data/q100_train.csv')
#     print(data)
#     print(data.isnull().sum())

#     make_drop_data(data)


# data= pd.read_csv('./data/q100_drop.csv')
# print(data.isnull().sum())
# print(data)

# items= data['class'].unique()
# encoder= LabelEncoder()
# encoder.fit(items)
# data['class']= encoder.transform(data['class'])
# # print(encoder.classes_)
# # print(encoder.inverse_transform(data['class']))
# print(data['class'])

# data.to_csv('/opt/ml/test_code/data/q100_num.csv', index= False)



# data= pd.read_csv('./data/q100_num.csv')
# data.rename(columns={'Unnamed: 0' : 'idx'}, inplace= True)

# print(data)
# train_df, test_df= train_test_split(data, test_size= 0.2, stratify= data['class'],random_state= 42)

# train_df.to_csv('/opt/ml/test_code/data/train.csv', index= False)
# test_df.to_csv('/opt/ml/test_code/data/test.csv', index= False)

train= pd.read_csv('./data/train.csv')
test= pd.read_csv('./data/test.csv')

print(train.groupby(train['class']).count())
print(test.groupby(test['class']).count())

sub= pd.read_csv('./prediction/submission0.csv')
print(sub.groupby(sub['pred_label']).count())




"""
    RE 흐름에 대한 Remind..
    일단 train.csv가 주어진다..!
    그거에서.. [SUB] [OBJ] Entity를 넣고자하는 듯..우리가.. 
    그래서 tokenize를 해줌..! 그리고 모델에 넣기전 dataset으로 다시 가공을 한 번 더 해주나..? 그건 잘 기억이 안남..
    모델에 넣음..! 그러면 우리가 cls token 기준으로..! 분류를 하는 느낌임..!
    pkl 파일을 만든다..!
"""