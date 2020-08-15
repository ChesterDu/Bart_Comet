import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from transformers import BartTokenizer, BartModel, BartForConditionalGeneration
from bart_evaluate import combine_into_words
import pickle
from data import tokenizer

eval_batch_size = 64
device = torch.device("cuda")


def train(model,dataset,optimizer,log_path, best_model_pth,batch_size = 16, num_accumulation = 4,epoch_num = 10):
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    iter_num = len(train_dataset) // batch_size
    best_ppl = 1000000
    step_count = 0
    with open(log_path,'w') as fout:
        a = 2

    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for iter in tqdm(range(iter_num)):
            input_id, input_mask, output_id, output_mask = gen_batched_data(batch_size, iter, train_dataset)
            bsz = input_id.shape[0]
            logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

            out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
            out = F.log_softmax(out)

            target = output_id[:, 1:].contiguous().reshape(-1)

            loss = F.nll_loss(out,target, reduction='none').view(bsz,-1)
            loss = (loss * output_mask[:,1:].float()).sum(1)
            length = output_mask.float().sum(1)
            loss = (loss/length).sum()/bsz

            loss.backward()

            epoch_loss += loss.item()

            if (iter + 1) % num_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1
                if step_count % 500 == 0:
                    train_loss = epoch_loss/(iter + 1)
                    test_perplexity = eval_model(model,test_dataset)
                    log = {"Step:":step_count,"Train Loss":train_loss,"Test Perplexity":test_perplexity}
                    log2file(log,log_path)
                    for key in log.keys():
                        print("{}: {}".format(key,log[key]))

            
            # break
        test_perplexity = eval_model(model,test_dataset)
        print("================Epoch %d===================="%(epoch))
        print("Training loss: %f"%(epoch_loss/iter_num))
        print("Test Perplexity: %f"%(test_perplexity))
        sample(model,test_dataset,10)
        if test_perplexity < best_ppl:
            best_ppl = test_perplexity
            torch.save(model.state_dict(), open(best_model_pth,"wb"))
            print("best model saved")
        


  
def log2file(log,log_path):
  with open(log_path,'a') as fout:
    for key in log.keys():
      fout.write("{}: {}\n".format(key,log[key]))
    
    fout.write("===================================\n")



def gen_batched_data(batch_size, iter, dataset, PAD_IDX = 1):
    st = iter * batch_size
    ed = min([(iter+1) * batch_size, len(dataset)])
    batched_data = dataset[st:ed]

    # max_input_len = max([len(data["input_ids"]) for data in batched_data])
    # max_output_len = max([len(data["output_ids"]) for data in batched_data])
    max_input_len = max([len(data[0]) for data in batched_data])
    max_output_len = max([len(data[1]) for data in batched_data])

    batched_input_id = []
    batched_output_id = []

    for data in batched_data:
        input_id,output_id = data
        input_id += [PAD_IDX] * (max_input_len - len(input_id))
        output_id += [PAD_IDX] * (max_output_len - len(output_id))
        # loss_mask += [0.0] * (max_output_len - len(loss_mask))
        
        batched_input_id.append(input_id)
        batched_output_id.append(output_id)
        # batched_loss_mask.append(loss_mask)

    batched_input_id = torch.LongTensor(batched_input_id).to(device)
    batched_output_id = torch.LongTensor(batched_output_id).to(device)
    # batched_loss_mask = torch.ByteStorage(batched_loss_mask).to(device)
    batched_input_mask = batched_input_id != PAD_IDX
    batched_output_mask = batched_output_id != PAD_IDX

    return batched_input_id, batched_input_mask, batched_output_id, batched_output_mask


def eval_model(model,eval_dataset):
    eval_iter_num = len(eval_dataset) // eval_batch_size
    model.eval()
    perplexity = 0
    for iter in range(eval_iter_num):
        with torch.no_grad():
            input_id, input_mask, output_id, output_mask = gen_batched_data(eval_batch_size, iter, eval_dataset)
            logits = model(input_ids = input_id, decoder_input_ids = output_id, labels = output_id)[1]

            out = logits[:, :-1, :].contiguous().reshape(-1,logits.shape[-1])
            out = F.log_softmax(out)
            # print(out.shape)
            target = output_id[:, 1:].contiguous().reshape(-1)
            # print(target.shape)

            loss = F.nll_loss(out,target, reduction='none').view(eval_batch_size,-1)
            loss = (loss * output_mask[:,1:].float()).sum(1)
            length = output_mask.float().sum(1)
            loss = (loss/length).sum()/eval_batch_size

            perplexity += loss.item()

    return np.exp(perplexity / eval_iter_num)



def sample(model,test_dataset,sample_num = 5):
    input_ids = gen_batched_data(sample_num, 0, test_dataset)[0]
    output_ids = model.generate(input_ids=input_ids, max_length=200,do_sample=False)
    for i in range(output_ids.shape[0]): #  3 output sequences were generated
        print('Generated {}: {}'.format(i,tokenizer.decode(output_ids[i], skip_special_tokens=True)))


