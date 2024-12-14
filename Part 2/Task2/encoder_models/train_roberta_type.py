import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaModel
from torch import nn
from tqdm import tqdm
from utils_type import main

class RobertaClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(RobertaClassifier, self).__init__()
        self.bert = RobertaModel.from_pretrained("FacebookAI/roberta-base")

        # Regularization
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(768, 3)
        self.relu = nn.ReLU() # Activation function: ReLU

    def forward(self, input_id, mask):
        #print(input_id, mask)
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

if __name__ == "__main__":

    """ 
    Select the correct dataset:
    - segments_types_final.xlsx: training on the raw data
    - segments_types_balanced.xlsx: training on the balanced data
    
    """
        
    #datafile = '..\..\segments_types_balanced.xlsx'
    datafile = '..\..\segments_types_final.xlsx'

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    labels = {0:0,
                1:1, 2:2}
    model = RobertaClassifier()
    name_model = 'Roberta' 

    if 'balance'in datafile:
        name_model += '_2'
    
    torch.save(model, name_model+'_base.pth')
    main(datafile, tokenizer, labels, name_model)
    
