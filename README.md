# Online Appendix of the paper "Locating Requirements in Backlog Items: Content Analysis and Experiments with Large Language Models"

In the paper, we examine how requirements are documented in backlogs stored in issue tracking systems and explore automated techniques to identify and classify requirements backlog items.
The appendix contains two main folders, each related to one research question.

## Folder: Part 1
This folder contains material related to RQ1.
The notebook `get_results.ipynb` presents the results based on the datasets produced by the files in the subfolder `tagged data`. This notebook generates the following files:
 `RQ1.1.xlsx`, `RQ1.2a.xlsx`, `RQ1.2b.xlsx`,  `RQ1.2c.xlsx`, `RQ1.3.xlsx`. 

- `RQ1.1.xlsx` presents the raw data of table 3
- `RQ1.2a.xlsx` contains the data of table 4
- `RQ1.2a.xlsx` contains the data of table 5
- `RQ1.2a.xlsx` contains the data of table 6
- `RQ1.3.xlsx` reports the raw data of table 7

This notebook utilizes the file `two_codes_per_req - OSS.xlsx` to produce one of the result tables. The Excel file contains the number of text segments tagged with two different codes/categories (per project).

### Subfolder “Coding guidelines.
The file `scheme issue requirements.pdf` defines the tagging guidelines we used for annotating the datasets.

### Subfolder “Tagged data.”
The script `merge_nvivo_datasets.py` combines the Nvivo files in the subfolder “Tagged data/Category per item” and creates an Excel file that indicates which categories occur per item (0 or 1).
The script `link_codes_to_df.py` uses the file exported by `merge_nvivo_datasets.py` to indicate per item whether a category occurs more than once.

#### Subsubfolder “Nvivo_extracted_v2”.
For each of the OSS projects, we include an exported Excel file from Nvivo with the tagged categories for each item (indicates only 1 or 0).

#### Subsubfolder “original_samples”.
For each of the OSS projects, we add the original sample items, including data on issue type and creation data.

#### Subsubfolder “raw tags”.
For each combination (type x granularity) we tagged (see the schema in the `coding schema` folder), we attach an exported txt file from Nvivo with the tagged text. There are 10 files (for each of the combinations + a file for the motivation tags), and each file contains the tagged text of the open source projects. For confidentiality reasons, we cannot share the tagged text of the proprietary projects.


## Folder: Part 2
This folder consists of material related to RQ2. This folder contains three subfolders.

### Main folder.
The main folder contains `convert_items_type.ipynb`, `label_data.py` and the `Figure A - example segmentation.pdf`, 
The pdf file `Figure A - example segmentation.pdf` provides an example of how the segmentation process is performed.
The file `label_data.py` uses the data from part 1 (the raw tags and original samples), segments the items and links the raw tags to each segment. This results in a file `segments_final.xlsx`, in which each row represents a segment with columns indicating whether it contains a particular tag (1 or 0 per category).
The file `convert_items_type.ipynb` converts the multiple tag columns to a single tag column for the classification task. It can export it to an Excel file named: `segments_types_final.xlsx`.

### Subfolder: balance_data
This subfolder contains the file `sample_data.ipynb`. This file uses the files `segments_final.xlsx` and `segments_types_final.xlsx` in the main directory and creates two balanced datasets `segments_balanced.xlsx` and `segments_types_balanced.xlsx` for identifying and classifying requirements, respectively.


### Subsubfolder: Task1
This subfolder contains the material related to Task 1. It contains two subfolders `decoder_models` and `encoder_models`.

#### Main Folder
The root folder contains `Figure B - heatmap.pdf`, `RQ2-1-precision-recall.pdf` (Figure 5) and `get_req_results.ipynb`. 
The file `get_req_results.ipynb` extracts the performance data from the `decoder_models` and `encoder_models` subfolders and produces the table results plus figures for Task1. To accomplish this, it also performs the hypothesis tests using the Python package SciPy. 

#### Subsubfolder: decoder_models
This subfolder contains:
- `gpt_f1.xlxs`: raw F1 score per model per project
- `gpt_precision.xlsx`: raw precision score per model per project
- `gpt_recall.xlsx`: raw recall score per model per project
- `prompt_chatgpt.txt`: prompt used for ChatGPT 
- `prompt.txt`:  prompt used for Mistral and Llama

#### Subfolder: encoder_models
The root folder contains the files `train_bert.py`, `train_roberta.py` and `utils.py`. The file `train_bert.py` executes the training for BERT and BERT_2. The file `train_roberta.py` performs the training for RoBERTa and RoBERTa_2. Both files use the file `utils.py` to use the training functions.

- Subfolder BERT: this folder contains the performance of BERT per project: Accuracy, Precision, Recall, TP, FP, FN
- Subfolder BERT_2: this folder contains the performance of BERT_2 by project: Accuracy, Precision, Recall, TP, FP, FN
- Subfolder Roberta: this folder contains the performance of RoBERTa by project: Accuracy, Precision, Recall, TP, FP, FN
- Subfolder Roberta_2: this folder contains the performance of RoBERTa_2 by project: Accuracy, Precision, Recall, TP, FP, FN


### subfolder: Task2
This subfolder consists of the materials related to Task2. It contains two subfolders `decoder_models` and `encoder_models`.

#### Main folder
The root folder contains `Figure C - heatmap.pdf`, `RQ2-2-precision-recall.pdf` (Figure 5) and `get_req_results_type.ipynb`. 
The file `get_req_results_type.ipynb` extracts the performance data from the `decoder_models` and `encoder_models` subfolders, and produces the table results plus figures for Task2. To accomplish this, it also performs hypothesis testing using the Python package SciPy. 

#### Subsubfolder: decoder_models
This subfolder contains:
- `gpt_f1_types.xlxs`: raw F1 score per model per project
- `gpt_precision_types.xlsx`: raw precision score per model per project
- `gpt_recall_types.xlsx`: raw recall score per model per project
- `prompt_type_chatgpt.txt`: prompt used for ChatGPT 
- `prompt_type.txt`:  prompt used for Mistral and Llama

#### Subsubfolder: encoder_models
The root directory contains the files `train_bert_type.py`, `train_roberta_type.py` and `utils_type.py`. The file `train_bert_type.py` performs the training for BERT and BERT_2. The file `train_roberta_type.py` performs the training for RoBERTa and RoBERTa_2. Both files use the file `utils_type.py` to use the training functions.
- Subfolder BERT: this folder contains the performance of BERT per project: Accuracy, Precision, Recall, TP, FP, FN
- Subfolder BERT_2: this folder contains the performance of BERT_2 by project: Accuracy, Precision, Recall, TP, FP, FN
- Subfolder Roberta: this folder contains the performance of RoBERTa by project: Accuracy, Precision, Recall, TP, FP, FN
- Subfolder Roberta_2: this folder contains the performance of RoBERTa_2 by project: Accuracy, Precision, Recall, TP, FP, FN



## Reference to data
L. Montgomery, C. Lüders, & W. Maalej (2022, May). An alternative issue tracking dataset of public jira repositories. In Proceedings of the 19th International Conference on Mining Software Repositories (pp. 73-77).

V. Tawosi, A. Al-Subaihin, R. Moussa, & F. Sarro (2022, May). A versatile dataset of agile open source software projects. In Proceedings of the 19th International Conference on Mining Software Repositories (pp. 707-711).


## Reference
A.T. van Can & F. Dalpiaz (2024). Locating Requirements in Backlog Items: Content Analysis and Experiments with Large Language Models. Information and Software
Technology 