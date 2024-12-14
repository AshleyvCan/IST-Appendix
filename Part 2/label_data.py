import codecs
import pandas as pd
import os
import re
import numpy as np
import string
import math
import spacy
import string

# Open txt files with tagged text
def open_codes(file_name):
    with codecs.open(file_name,  'r', encoding="utf-16") as f:
        lines = f.read()

    return lines

# Remove URLs from string
def remove_url(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text_no_urls = url_pattern.sub('', text)

    return text_no_urls

# Verify whether the segments in the enumeration contain a verb
def check_bullets(text, punc, nlp):
    doc = nlp(text.replace(punc, '').strip())

    if len([token.lemma_ for token in doc if token.pos_ == "VERB"]) > 0:
        return 1
    return 0

# For a given enumartion, split the text by line
def split_string(input_string, nlp):
    
    lines = input_string.splitlines()
    first_line = ''
    patterns = {
        '\n   *': r'\n {3}\*(?!\*)',  # bullets'*' with three space
        '\n  *': r'\n {2}\*(?!\*)',    # bullets '*' with two space
        '\n -': r'\n -(?![\-])',      # bullets '-' with space
        '\n |': r'\n \|(?![|])',      # bullets '|' with space
        '\n #': r'\n \#(?![#])',      # bullets '#' with space
        '\n *': r'\n \*(?![*])',      # bullets '*' with space
        '\n*': r'(?<!:)\n\*(?!\*)',   # bullets '*'
        '\n-': r'(?<!:)\n-(?![\-])',  # bullets '-'
        '\n|': r'(?<!:)\n\|(?![|])',  # bullets '|'
        '\n#': r'(?<!:)\n\#(?![#])',   # bullets '#'
        '\n\d+\.': r'\n\d+\.',
        '|*': r'\|\*(?!\*)',
        '** +': r'\*\*\s+\+'    
    }

    present_patterns = {p: regex for p, regex in patterns.items() if re.search(regex, input_string)}
    
    if len(lines) > 1 and len(lines[1].strip()) > 0 and lines[1].strip()[-1] == ':': 
        if len(lines[0].strip()) > 0 and lines[0].strip()[0] in ['*', '-', '#', '|']  and lines[1].strip()[0] in ['*', '-', '#', '|'] and lines[1].strip()[:2] != '**': 
            first_line = lines[:2]
        else:    
            first_line = '\n'.join(lines[:2])
        lines = lines[2:]
        
    elif len(lines) > 0 and len(lines[0].strip()) > 0 and lines[0].strip()[-1] == ':' and len(lines[0].split('.')) < 2: 
        first_line = lines[0]
        lines = lines[1:]

    if not present_patterns:
        return [input_string]

    split_result = ['\n'.join(lines)]

    for pattern, regex in present_patterns.items():
        if sum([check_bullets(line, pattern, nlp) for line in lines]) > 0:
            temp_result = []

            for s in split_result:
                temp_result.extend(re.split(regex, s))
            split_result = temp_result

    split_result = [s for s in split_result if s]
    
    if len(split_result)> 0:
        if isinstance(first_line, list):
            split_result[0] = first_line[1] + '\n'+ split_result[0]
            split_result = [first_line[0]] + split_result
        else:
            split_result[0] = first_line + '\n'+ split_result[0]
    
    return split_result


# For a given tagged text (req), locate the correct segment.
def find_correct_item(req, items, k):
    best_match = None
    best_score = float('inf') 

    for index, item in items.iterrows():
        description = item['text_processed']
        
        if isinstance(description, str):
            pos = description.find(req)
            
            if pos != -1:
                last_end = max(description.rfind('\n', 0, pos), description.rfind('.', 0, pos))    
                score = len(description)-last_end - len(req)
                  
                if score < best_score and item[k] != 1:
                    best_score = score
                    best_match = index
                    best_key = item['id']
                elif score == best_score and item['id'] != best_key and item[k] != 1:
                    if isinstance(best_match, list): 
                        best_match.append(index)
                    else: 
                        best_match = [best_match, index]
        
    if isinstance(best_match, list): return best_match[0]
    else: return best_match


# Split item in segments
def split_items(df, project_name, nlp, attach_header = False, limit= 2000):
    candidate_reqs = []
    for i, row in df.iterrows():
        content = [row['description'], row['summary']]

        header = ''
        for text in content:
            
            if not pd.isna(text):
                text = remove_url(text)
                for p1 in re.split(r'\n\s*\n', text):
                    puncs = ['*', '-', '|', '#'] 
                    candidate_req = split_string(p1, nlp)
                    for p2 in candidate_req:
                        if attach_header and len(p2) > 0 and p2.strip()[0] == '*' and p2.strip()[-1] == '*': 
                            header = p2
                        elif not all(char in string.punctuation for char in p2):
                            if len(header) > 0:
                                p2 = header + '\n\n' + p2
                                header = ''
                            if len(p2) > limit:
                                
                                
                                for p3 in p2.split('\n'):
                                    candidate_reqs.append({'id': row['id'],'project': project_name,'text': p3})
                                    #print('split')
                            else:
                                candidate_reqs.append({'id': row['id'],'project': project_name,'text': p2})
                    
    return candidate_reqs

# Link each tagged text to the correct segment
def link_reqs_to_items(candidate_reqs, codes_2, missed_reqs, project_name):
    df = pd.DataFrame(candidate_reqs)

    missed_reqs[project_name] = {}
    df['text_processed'] = df['text'].str.replace('_x000D_', ' ').str.replace('\n',' ').str.replace('\s+', ' ', regex = True)

    for k in codes_2.keys():
        df[k] = len(df) * 0
        missed_reqs[project_name][k] = []
        if project_name in codes_2[k].keys():

            for code in codes_2[k][project_name]:
                text = remove_url(code)
                text = re.sub('\s+', ' ', text.replace('_x000D_', ' ').replace('\r', ' ').replace('\t', ' ').replace('\n', ' ').strip())

                if len(df.loc[df['text_processed'].str.contains(text, regex = False)]) == 0:
                    missed_reqs[project_name][k].append(text)
                
                elif text.replace(' ', '') not in ['', '\n','\r'] and len(text) > 0: 
                    if len(df[(df['text_processed'].str.contains(text, regex = False)) & (df[k] == 0)]) > 1:
                        i_loc = find_correct_item(text, df[df['text_processed'].str.contains(text, regex = False)], k)
                        df.loc[i_loc, k] += 1
                       
                    else:
                        df.loc[df['text_processed'].str.contains(text, regex = False),k]+= 1

    return df.to_dict(orient= 'records'), missed_reqs

# Main function to link tags to item segments
def main(codes):
    project_names = [codes['medium_user'][i].split('Reference')[0].split('> -')[0].replace('\\', '') for i in range(len(codes['medium_user']))]

    codes_2 = {k: {v[i].split('Reference')[0].split('> -')[0].replace('\\', ''): [ref.split('\r\n\r\n')[1].strip(' .,+!?/\"#*-')  for ref in v[i].split('Reference')[1:]]
                        for i in range(len(v))}
                        for k,v in codes.items()}
    nlp = spacy.load("en_core_web_md")    
    
    data_labelled = []
    all_missed_reqs = {}
    for i in range(len(project_names)):
        project_name = project_names[i]
        print(project_name)
        df = pd.read_excel(dir_backlogs + project_name + '.xlsx')

        missed_reqs = {}
        candidate_reqs = split_items(df, project_name, nlp)
        df_subset, missed_reqs = link_reqs_to_items(candidate_reqs, codes_2, missed_reqs, project_name)
        data_labelled.extend(df_subset)
        all_missed_reqs.update(missed_reqs)

    print(all_missed_reqs)
    df_labelled = pd.DataFrame(data_labelled)
    df_labelled.to_excel('segments_final.xlsx')



if __name__ == "__main__":
    dir = '../Part 1/tagged data/tags/'
    dir_backlogs = '../Part 1/tagged data/original_samples/'

    codes = {'high_user': open_codes(dir+'high_user.txt').split('<Files\\')[1:],
            'high_system': open_codes(dir+'high_system.txt').split('<Files\\')[1:],
            'high_nfr': open_codes(dir+'high_nfr.txt').split('<Files\\')[1:],
            'medium_user': open_codes(dir+'medium_user.txt').split('<Files\\')[1:],
            'medium_system': open_codes(dir+'medium_system.txt').split('<Files\\')[1:],
            'medium_nfr': open_codes(dir+'medium_nfr.txt').split('<Files\\')[1:],
            'low_user': open_codes(dir+'low_user.txt').split('<Files\\')[1:],
            'low_system': open_codes(dir+'low_system.txt').split('<Files\\')[1:],
            'low_nfr': open_codes(dir+'low_nfr.txt').split('<Files\\')[1:]
            }
    
    main(codes)