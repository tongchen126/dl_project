import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from gensim.models import word2vec
import re

def pd_load_csv(file,cols=None):
    with open(file,'r') as f:
        return pd.read_csv(f,usecols=cols)

def pd_load_json(file,cols=None):
    with open(file,'r') as f:
        return pd.read_json(f, orient="records", lines=True)

def construct_dataset(business:pd.DataFrame,review:pd.DataFrame,covid:pd.DataFrame):
    business_covid = pd.merge(business,covid,on='business_id')
    return pd.merge(business_covid,review,on='business_id')

def get_word2vec(x,vector_size=250,save_path='model/yelp/w2v_yelp.model'):
    model = word2vec.Word2Vec(x, vector_size=vector_size, window=5, min_count=5, workers=12)
    if save_path:
        model.save(save_path)
    return model

class YelpDataset(Dataset):
    USECOLS = [
        'city', 'state', 'latitude', 'longitude', 'stars_x', 'review_count', 'categories',
        'stars_y', 'date', 'text', 'useful',
        'highlights', 'delivery or takeout', 'Grubhub enabled', 'Call To Action enabled', 'Request a Quote Enabled', 'Covid Banner', 'Virtual Services Offered'
    ]
    CLASS_COLS = []
    INT_COLS = ['review_count','stars_x','stars_y','useful']
    TEXT_COLS = ['categories','text']
    COVID_COLS = ['highlights', 'delivery or takeout', 'Grubhub enabled', 'Call To Action enabled', 'Request a Quote Enabled', 'Covid Banner', 'Virtual Services Offered']

    def __init__(self,data_path):
        self.raw_data = pd_load_csv(data_path,cols=self.USECOLS)
        self.data = {}
        self.word2vec_dataset = []

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, item):
        if item in self.data:
            return self.data[item]
        int_array = []
        class_array = []
        text_array = []
        covid_array = []
        row = pd.Series(self.raw_data.iloc[item])

        for col in self.INT_COLS:
            int_array.append(int(row[col]))

        for col in self.CLASS_COLS:
            class_array.append(int(row[col]))

        for col in self.TEXT_COLS:
            text_array.append(row[col])

        for col in self.COVID_COLS:
            col_value = row[col]
            col_value = str(col_value).upper() == 'FALSE'
            covid_result = 0 if col_value else 1
            covid_array.append(covid_result)
        covid_array = np.sum(covid_array)

        self.data[item] = (int_array,class_array,text_array,covid_array)

        return self.data[item]

    def prepare_word2vec_data(self,maxlen=5e6):
        if len(self.word2vec_dataset):
            return self.word2vec_dataset
        for i in range(len(self)):
            _,_,texts,_ = self[i]
            self.word2vec_dataset.extend([list(filter(None,re.split(' |;|,|\.|&',line.strip('\n')))) for line in texts])
            print(len(self.word2vec_dataset))
            if len(self.word2vec_dataset) > maxlen:
                break
        return self.word2vec_dataset

if __name__ == '__main__':
    dataset = YelpDataset('/root/Downloads/yelp/kaggle/business_review_covid.csv')
    w2v_data = dataset.prepare_word2vec_data()
    get_word2vec(w2v_data)
    print('end')