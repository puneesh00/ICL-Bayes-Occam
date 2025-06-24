import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import log_results, save_model
from utils_data import generate_evaluation_seqs, run_eval
from utils_data import MarkovChainGenerator, MarkovChainEvaluator

from gpt import GPT, gpt, create_padding_mask
from minigpt_utils import CfgNode


def train_transformer_gpt(transformer, steps, vocab_size, max_transitions, log_interval, save_interval, 
                            sequence_length, test_seq_len, lr, batch_size, order_list, 
                            test_order_list, t_gram_list, device, 
                            log_files,tgram_string,save_path):
    
    base_seed=0

    num_tm_eval = 10 #number of MCs used for evaluation

    test_eval  =  generate_evaluation_seqs(order_list=test_order_list, 
                                           num_tm=num_tm_eval, vocab_size=vocab_size, 
                                           max_transitions=max_transitions, 
                                           t_gram_list=t_gram_list, 
                                           test_seq_len=test_seq_len)

    
    

    train_config = CfgNode()
    train_config.learning_rate = lr 
    train_config.weight_decay = 1e-3 
    train_config.betas = (0.9, 0.95)
    optimizer = transformer.configure_optimizers(train_config)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000, 6000], gamma=0.6)

    for step in range(steps):

        idx = np.random.choice(len(order_list), batch_size)
        ord = [order_list[i] for i in idx]
        
        sequences=[]
        datas, targets = [], []

        #create fresh batch of sequences
        for i in range(batch_size):
            seed = base_seed + step + i

            generator = MarkovChainGenerator(vocab_size, ord[i], max_transitions, seed)

            sequence = generator.generate_sequence(sequence_length)
            data, target = generator.generate_data_target_pairs(sequence, len(sequence))

            datas.append(data)
            targets.append(target)
            sequences.append(sequence)
            
        datas = np.squeeze(np.asarray(datas))
        targets = np.squeeze(np.asarray(targets))
        
        #Convert data and targets to tensors
        data_tensor = torch.tensor(datas, dtype=torch.long).to(device)
        target_tensor = torch.tensor(targets, dtype=torch.long).to(device)
        
        transformer.train()
        l = 0
        
        optimizer.zero_grad()
        batch_size, seq_len = data_tensor.size()

        # Forward pass
        logits, loss = transformer(data_tensor, targets=target_tensor) 
        
        loss.backward()
        
        optimizer.step()
        l+=loss.item()
        scheduler.step()    

        # Calculate test loss (KL divergence)   
        l = l / (len(datas) / batch_size)
        if (step) % log_interval == 0:  # Evaluate every log_epoch epochs
            transformer.eval()


            with torch.no_grad():

                test_res = run_eval(test_order_list=test_order_list, test_seqs_list=test_eval["seqs"], 
                                    device=device, 
                                    transformer=transformer,
                                    t_gram_stats=test_eval["tgram_stats"],t_grams=t_gram_list, 
                                    step=step) #test_evaluation
            
            log_results(log_files, step, l, test_res, tgram_string)

        if step%save_interval==0 or step==steps-1:
            save_model(transformer, step=step, save_path=save_path)
    for ord in order_list:
        log_files[ord].close()     