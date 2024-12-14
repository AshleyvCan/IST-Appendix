import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, labels):
        self.labels = [labels[label] for label in df['label']]
        # Tokenize all segments
        self.texts = [tokenizer(text, padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)
    
    def classes(self):
        return self.labels

    def get_batch_labels(self, index):
        return np.array(self.labels[index])

    def get_batch_texts(self, index):
        return self.texts[index]

    def __getitem__(self, index):

        batch_texts = self.get_batch_texts(index)
        batch_y = self.get_batch_labels(index)

        return batch_texts, batch_y


def export_model(model, filename):
    return torch.save(model, filename)
    
def import_model(filename):
    return torch.load(filename)


# Train the given model on train_data with batch_size 16
def train(model, train_data, val_data, l_rate, epochs, tokenizer, labels, name_model, dir_model):

    train_set = Dataset(train_data, tokenizer, labels)
    val_set = Dataset(val_data, tokenizer, labels)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    optimizer = Adam(model.parameters(), lr= l_rate)

    acc_train_all_epochs = []
    loss_train_all_epochs = []
    acc_val_all_epochs = []
    loss_val_all_epochs = []

    early_stop_thresh = 4
    best_accuracy = -1
    best_epoch = -1
    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            acc_train_all_epochs.append(total_acc_train/ len(train_data))
            loss_train_all_epochs.append(total_loss_train / len(train_data))

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            val_acc = total_acc_val/ len(val_data)
            acc_val_all_epochs.append(val_acc)
            loss_val_all_epochs.append(total_loss_val/ len(val_data))
            
            if val_acc > best_accuracy:
                print(val_acc, best_accuracy)
                best_accuracy = val_acc
                best_epoch = epoch_num
                export_model(model, dir_model+'/'+name_model+"_best_model.pth")
            elif epoch_num - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch_num)
                break  


            print(f'Epoch: {epoch_num + 1}, Train Loss: {total_loss_train / len(train_data): .4f}, Train Accuracy: {total_acc_train / len(train_data): .4f}')
            print(f'Val Loss: {total_loss_val / len(val_data): .4f} | Val Accuracy: {total_acc_val / len(val_data): .4f}')
    return pd.DataFrame({'train_accuracy': acc_train_all_epochs, 'train_loss': loss_train_all_epochs, 'val_accuracy': acc_val_all_epochs, 'val_loss':loss_val_all_epochs})

# Identify whether a given prediction is correct (TP), or whether it is an FP or FN.
def performance(pred, real, unique_classes, TP, FP, FN):
    for c in unique_classes:
        for i in range(len(pred)):
            if pred[i] == c and real[i] == c:
                TP[c] += 1
            elif pred[i] == c and real[i] != c:
                FP[c] += 1
            elif pred[i] != c and real[i] == c:
                FN[c] += 1   
    return TP, FP, FN

# Export the performance to a txt file.
def export_performance(file_name, unique_classes, total_acc_test, test_data, TP, FP, FN):

    with open(file_name + '.txt', 'w') as text_file:
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}', file=text_file)
        print('Precision:', {c:TP[c]/(TP[c] + FP[c]) if (TP[c] + FP[c])> 0 else 0 for c in unique_classes}, file=text_file)
        print('Recall:', {c:TP[c]/(TP[c] + FN[c]) if (TP[c] + FN[c])> 0 else 0  for c in unique_classes}, file=text_file)
        print('TP:',TP , file=text_file)
        print('FP:',FP , file=text_file)
        print('FN:',FN , file=text_file)

# Evaluate a given model on a test set
def evaluate(model, test_data, tokenizer, labels, name_model, dir_model):
    unique_classes = set(test_data['label'])
    test_set = Dataset(test_data, tokenizer, labels)

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model = model.cuda()

    total_acc_test = 0
    TP = {c:0 for c in unique_classes}
    FP = {c:0 for c in unique_classes}
    FN = {c:0 for c in unique_classes}

    with torch.no_grad():
        i = 0
        for test_input, test_label in test_dataloader:
              i = i +1
              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)
              output = model(input_id, mask)
              i = i+ 1
              test_acc = (output.argmax(dim=1) == test_label).sum().item()
              TP, FP, FN = performance(output.argmax(dim=1), test_label, unique_classes, TP, FP, FN)
              total_acc_test += test_acc
    export_performance(dir_model+'/'+name_model, unique_classes, total_acc_test, test_data, TP, FP, FN)


# Import the base model, read the dataset, and perform a project-fold cross-validation on the base model.
def main(datapath, tokenizer, labels, name_model):
    folder_name = name_model

    folder_path = os.path.join(os.getcwd(), folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df = pd.read_excel(datapath)

    if name_model.split('_')[-1] != '2':
        df['sum_req'] = df[df.columns[-9:]].sum(axis=1)
        df['label'] = [1 if l > 0 else 0 for l in df['sum_req']]

    df = df[(~df['text'].isna())]
    all_val_projects = [ 'Cost_Management', 'Jira_Performance_Testing_Tools', 'Lyrasis Dura Cloud', 'Network_Observability', 'OpenShift_UX_Product_Design', 'Qt_Design_Studio','Red_Hat_Developer_Website_v2']

    test_project = '' # leave oneset out 
    for val_project in all_val_projects: 

        # (re)load base model
        model_base = torch.load(name_model+'_base.pth')
        
        #Split train and val set
        df_train = df.loc[~df['project'].isin([val_project, test_project]), ['text', 'label']]
        df_val = df.loc[df['project'] == val_project, ['text', 'label']]
        
        # Run model
        EPOCHS = 10
        LR = 1e-5
        print(name_model+val_project)            
        epoch_performance = train(model_base, df_train, df_val, LR, EPOCHS,tokenizer, labels,name_model+ '_'+ val_project, folder_path)
        epoch_performance.to_excel(folder_path+'/'+name_model+ '_'+ val_project + '_epochs.xlsx')


    
    
