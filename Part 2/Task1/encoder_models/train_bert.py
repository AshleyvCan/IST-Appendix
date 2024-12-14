import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from tqdm import tqdm
from utils import main

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # Regularization
        self.dropout = nn.Dropout(dropout) 
        
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU() # Activation function: ReLU

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

if __name__ == "__main__":
    
    """ 
    Select the correct dataset:
    - segments_final.xlsx: training on the raw data
    - segments_balanced.xlsx: training on the balanced data
    
    """

    #datafile = '..\..\segments_final.xlsx'
    datafile = '..\..\segments_balanced.xlsx'
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    labels = {0:0,
                1:1}
    model = BertClassifier()
    name_model = 'BERT' 

    if 'balance'in datafile:
        name_model += '_2'

    torch.save(model, name_model+'_base.pth')
    main(datafile, tokenizer, labels, name_model)
