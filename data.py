import torch
from transformers import BartTokenizer
import tqdm
from bart_evaluate import combine_into_words
import numpy as np

language_dict = {combine_into_words[k]:k for k in combine_into_words.keys()}
tokenizer = BartTokenizer.from_pretraine('facebook/bart-large')

mask = tokenizer.encode("<mask>")[1:-1][0]

def make_pretrain_dataset(setting,saved_data_pth = None,raw_data_pth = None,processed_data_pth = None):
    if processed_data_pth != None:
        dataset = torch.load(processed_data_pth)
        return dataset
    
    if setting.data_name == "graph":
        dataset = process_pretrain_graph_raw_data(raw_data_pth,saved_data_pth)
        return dataset
    
    if setting.data_name == "path":
        dataset = process_pretrain_path_raw_data(setting,raw_data_pth,saved_data_pth)
        return dataset


def make_finetune_dataset(saved_data = None):
    if saved_data != None:
        dataset = torch.load(saved_data)
        return dataset

def process_pretrain_graph_raw_data(raw_data_pth,processed_data_pth = None):
    raw_data = torch.load(raw_data_pth)     #list of tuples
    processed_dataset = []
    for tuple_path in tqdm(raw_data):
        input_lst = mege_to_sequence(tuple_path[0])
        output_lst = mege_to_sequence(tuple_path[1])
        input_ids = lst2ids(input_lst)
        output_ids = lst2ids(output_lst)
        processed_dataset.append((input_ids,output_ids))

    if processed_data_pth != None:
        torch.save(processed_dataset,processed_data_pth)

    return processed_dataset

def  process_pretrain_path_raw_data(setting,raw_data_pth,saved_data_pth):
    raw_data = torch.load(raw_data_pth)
    if setting.corruption == "none":
        processed_dataset = []
        for tuple_path in tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            input_ids = lst2ids(input_lst)
            output_ids = lst2ids(output_lst)
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset

    if setting.corruption == "deletion":
        ratio = float(setting.ratio)
        processed_dataset = []
        for tuple_path in tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            output_ids = lst2ids(output_lst)
            input_ids = ids_deletion(output_ids,ratio)
            
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset
        
        
    if setting.corruption == "masking":
        ratio = float(setting.ratio)
        processed_dataset = []
        for tuple_path in tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            output_ids = lst2ids(output_lst)
            input_ids = ids_masking(output_ids,ratio)
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset

    if setting.corruption == "infilling":
        processed_dataset = []
        for tuple_path in tqdm(raw_data):
            input_lst = mege_to_sequence(tuple_path)
            output_lst = mege_to_sequence(tuple_path)
            output_ids = lst2ids(output_lst)
            input_ids = ids_infilling(output_ids,setting.num_of_span,setting.lamda)
            processed_dataset.append((input_ids,output_ids))

        if saved_data_pth != None:
            torch.save(processed_dataset,saved_data_pth)

        return processed_dataset




def mege_to_sequence(lst_triples):
  seq = []
  for i in range(0,len(lst_triples)):
    seq.append(lst_triples[i][0])
    seq.append(language_dict[lst_triples[i][1]])
    if i == len(lst_triples) - 1:
      seq.append(lst_triples[i][2])
  return seq


def lst2ids(lst):
  ids = []
  for i,ents in enumerate(lst):
    if i != (len(lst) -1):
      ids += tokenizer.encode(ents + ' ')[1:-1]
      # ids += [1]
    else:
      ids += tokenizer.encode(ents)[1:-1]
  ids = [0] + ids + [2]

  return ids

def ids_deletion(ids,ratio):
  ids = ids[1:-1]

  deletion_number = int(len(ids) * ratio)

  for i in range(deletion_number):
      delete_id = np.random.randint(0,len(ids)-1)
      ids.pop(delete_id)

  ids = [0] + ids + [2]

  return ids

def ids_masking(ids,ratio):
  ids = ids[1:-1]

  masking_number = int(len(ids) * ratio)

  mask_ids = []
  for i in range(masking_number):
      mask_id = np.random.randint(0,len(ids)-1)
      while mask_id in mask_ids:
          mask_id = np.random.randint(0,len(ids)-1)
      ids[mask_id] = mask
      mask_ids.append(mask_id)

  ids = [0] + ids + [2]

  return ids

def ids_infilling(ids,num_of_span,lam):
    ids = ids[1:-1]

    span_len_lst = np.random.poisson(lam=lam, size=num_of_span)
    while sum(span_len_lst) >= len(ids):
        span_len_lst = np.random.poisson(lam=lam, size=num_of_span)

    orig_len_lst = np.random.rand(len(span_len_lst))
    ratio = (len(ids) - sum(span_len_lst)) / sum(orig_len_lst)
    orig_len_lst = orig_len_lst * ratio
    orig_len_lst = [int(x) for x in orig_len_lst]
    orig_len_lst.append((len(ids) - sum(span_len_lst)) - sum(orig_len_lst))

    ret_ids = []
    start_id = 0
    for i in range(len(span_len_lst)):
        span_len = span_len_lst[i]
        orig_len = orig_len_lst[i]
        ret_ids += ids[start_id:start_id + orig_len]
        ret_ids += [mask]
        start_id = start_id + span_len + orig_len

    ret_ids = [0] + ret_ids + [2]

    return ret_ids

    



