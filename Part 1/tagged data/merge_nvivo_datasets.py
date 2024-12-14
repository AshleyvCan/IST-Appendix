import codecs
import pandas as pd
import os
import re
import numpy as np
import string

# Combining the Nvivo exported files and creating an Excel file that indicates which categories occur per item (0 or 1).
def merge_and_filter_files(files, possible_categories, dir, dir_org):
    codes = []

    for f in files:
        print(f)
        df = pd.read_excel(dir + f)
        df = df[['Scope Item', 'In Folder']]
        
        df = df[df['Scope Item'] != 'Scope Item'].reset_index()
        df['In Folder'] = df['In Folder'].str.lower()

        df_org = pd.read_excel(dir_org + f)
        df_org['id'] = df_org['id'].astype(str)    

        start_indices= list(df[~df['Scope Item'].isna()].index)
        end_indices = list(df[~df['Scope Item'].isna()].index - 1)[1:] + [df.index[-1]]

        codes_all_items = []
        
        for i in range(len(start_indices)):
            
            start_index = start_indices[i]
            end_index = end_indices[i]
            
            item_df = df.iloc[start_index:end_index+1]
            item_id = df.iloc[start_index]['Scope Item']
            item_info = df_org.loc[df_org['id'] == item_id]
            item_codes = list(item_df[(item_df['Scope Item'].isna()) & (~item_df['In Folder'].isin(['summary','range item','description', 'title']))]['In Folder'])

            item_all_codes = {c:(1 if c in item_codes else 0) for c in possible_categories}
            item_all_codes['id'] = item_id
            item_all_codes['project_name'] = f[:-5]
 
            item_all_codes['summary'] = item_info['summary'].values[0]
            item_all_codes['description'] = item_info['description'].values[0]
            item_all_codes['issuetype'] = item_info['issuetype'].values[0] 
            
            codes_all_items.append(item_all_codes)
            
        codes.extend(codes_all_items)
    pd.DataFrame(codes).to_excel('categories_per_item.xlsx')

if __name__ == "__main__":
    dir = 'Nvivo_extracted_v2/'
    dir_org = 'original_samples/'

    files = [f for f in os.listdir(dir) if f[-4:] == 'xlsx']
    possible_categories = ['low_user', 'low_system', 'low_nfr','medium_user', 'medium_system', 'medium_nfr','high_user', 'high_system', 'high_nfr', 'motivation']

    merge_and_filter_files( files, possible_categories,dir, dir_org)