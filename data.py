import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import codecs

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    '''
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>. 
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    '''
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split('.')[0][7:]
        with codecs.open(filename, 'r', encoding='utf8') as f:
            articles[article_id] = f.read()
    return articles

def read_predictions_from_file(filename):
    '''
    Reader for the gold file and the template output file. 
    Return values are four arrays with article ids, labels 
    (or ? in the case of a template file), begin of a fragment, 
    end of a fragment. 
    '''
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, 'r') as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split('\t')
            articles_id.append(article_id)
            gold_labels.append((gold_label, int(span_start), int(span_end)))
    return articles_id, gold_labels

def label(text, gt_labels):
    tokens = []
    labels = []
    special_symbols = """!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~ \n\t\'\\"""
    sentence = []
    sent_labels = []
    word = ''
    inside = False
    word_start = 0
    for i in range(len(text)):
        if text[i] in special_symbols:
            if len(word) > 1:
                sentence.append(word)
                word = ''
                if inside:
                    sent_labels.append(1)
                else:
                    sent_labels.append(0)
                # if the sentence has ended
                if text[i] in "!.?\n" and (i < len(text) - 2 and not (text[i+1].islower() or text[i+2].islower())):
                    if len(sentence) > 1:
                        tokens.append(sentence)
                        if any(sent_labels):
                            labels.append(1)
                        else:
                            labels.append(0)
                        sentence = []
                        sent_labels = []
        else:
            if len(word) == 0:
                word_start = i
            word += text[i]
        if len(gt_labels) > 0:
            if i == gt_labels[0][1]:
                inside = True
            elif i == gt_labels[0][2] + 1:
                inside = False
                gt_labels.pop(0)
    return tokens, labels
    

def create_dataset(path_to_articles, path_to_labels):
    '''
    Creates the dataset from the files contained in 'datasets/train-articles/' folder
    
    texts : list, each represents one article and contains
    '''
    texts = []
    labels = []
    articles = read_articles_from_file_list(path_to_articles)
    article_names = list(articles.keys())
    prefix_lbl = path_to_labels + '/article'
    postfix_lbl = '.task-flc-tc.labels'
    for name in article_names:
        articles_id, gold_labels = read_predictions_from_file(prefix_lbl + name + postfix_lbl)
        gt_labels = []
        for i in range(len(gold_labels)):
            if gold_labels[i][0] == 'Loaded_Language':
                gt_labels.append(gold_labels[i])
        gt_labels.sort(key=lambda x: x[1])
        tokens, lbls = label(articles[name], gt_labels)
        texts.extend(tokens)
        labels.extend(lbls)
    
    return texts, labels, article_names


class ManipulationDataset(Dataset):
    def __init__(self, articles_dir, labels_dir, max_seq_len=50):
        self.articles_dir = articles_dir
        self.labels_dir = labels_dir
        self.texts, self.labels, self.article_names = create_dataset(self.articles_dir, self.labels_dir)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        tokenized = self.tokenizer.encode_plus(' '.join(self.texts[i]), None, add_special_tokens=True, 
                                               max_length=self.max_seq_len,pad_to_max_length=True, return_token_type_ids=True)
        inputs = torch.tensor(tokenized['input_ids'][:self.max_seq_len])
        mask = torch.tensor(tokenized['attention_mask'][:self.max_seq_len])
        lbls = [1, 0] if self.labels[i] == 0 else [0, 1]
        return inputs, mask, torch.tensor(self.labels[i]), torch.tensor(lbls)