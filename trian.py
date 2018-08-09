from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import wikiwords
import copy
import torch.optim.lr_scheduler as lr_scheduler
import pdb
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

import unicodedata

samples=np.load('samples.npy')
samples_dev=np.load('samples_dev.npy')
USE_CUDA=True


class MCQ_RNN(nn.Module): 
    def __init__(self,n_layers=1, dropout=0.4,training=True):
        super(MCQ_RNN, self).__init__()
        self.hidden_size=96
        self.glove_size = 300
        self.n_layers=n_layers
        self.vocab_size = 33382
        self.pos_vocab_size = 51
        self.pos_size = 12
        self.ner_vocab_size = 20
        self.ner_size = 8
        self.rel_vocab_size = 39
        self.rel_size = 10
        self.dropout=dropout
        self.training=training

        self.init_embeddings()
        self.init_rnns()
        self.init_linears()
        
        
    def init_embeddings(self):
    	
        #fill glove embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.glove_size, padding_idx=0)
        self.embedding.weight.data.fill_(0)
        self.embedding.weight.data[:2].normal_(0, 0.1)
        self.load_embeddings(vocab_tokens, './data/glove.840B.300d.txt')
    
        #pos
        self.pos_emb_func = nn.Embedding(self.pos_vocab_size, self.pos_size, padding_idx=0)
        self.pos_emb_func.weight.data.normal_(0, 0.1)
        #ner
        self.ner_emb_func = nn.Embedding(self.ner_vocab_size, self.ner_size, padding_idx=0)
        self.ner_emb_func.weight.data.normal_(0, 0.1)
        #rel
        self.rel_emb_func = nn.Embedding(self.rel_vocab_size, self.rel_size, padding_idx=0)
        self.rel_emb_func.weight.data.normal_(0, 0.1)
        
    def init_rnns(self):
        input_size=2 * self.glove_size +self.pos_size + self.ner_size + 5+ 2 * self.rel_size
        
        #input_size=2 * self.embedding_dim + self.pos_emb_dim + self.ner_emb_dim + 2 * self.rel_emb_dim
        self.passage_rnn =nn.LSTM(input_size, self.hidden_size,
                                      num_layers=self.n_layers,
                                      bidirectional=True) 
                                      
        input_size = self.glove_size + self.pos_size                            
        self.question_rnn=nn.LSTM(input_size, self.hidden_size,
                                      num_layers=self.n_layers,
                                      bidirectional=True)
                                      
        input_size = 3 * self.glove_size                             
        self.answer_rnn=nn.LSTM(input_size, self.hidden_size,
                                      num_layers=self.n_layers,
                                      bidirectional=True)
                                      
    def init_linears(self):
        self.double_hidden_size=2*self.hidden_size
        
        self.p_c_bilinear = nn.Linear(self.double_hidden_size, self.double_hidden_size)
        self.q_c_bilinear = nn.Linear(self.double_hidden_size, self.double_hidden_size)      
        
        self.linear_q=nn.Linear(self.double_hidden_size, 1)
        self.linear_c=nn.Linear(self.double_hidden_size, 1)
        
        self.linear1 = nn.Linear(self.glove_size, self.glove_size)
        self.linear2 = nn.Linear(self.glove_size, self.glove_size)
        self.linear3 = nn.Linear(self.glove_size, self.glove_size)

        self.bi_linear=nn.Linear(self.double_hidden_size, self.double_hidden_size)
        
    def forward(self, p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, p_q_relation, p_c_relation):
        	#pre_version
        passage = [p, p_pos, p_ner, p_mask, f_tensor, p_q_relation, p_c_relation]
        question = [q, q_pos, q_mask]
        answer = [c, c_mask]
    
        	#Input Layer
        w_q, q_info = self.InputLayer(question, flag='question')
        w_p, p_info = self.InputLayer(passage, flag='passage')
        w_a, a_info = self.InputLayer(answer, flag='answer')
    
        q_emb_glove = w_q[0]
        p_emb_glove = w_p[0]
        a_emb_glove = w_a
        
        p_q_weighted_emb = self.AttentionLayer(p_emb_glove, q_emb_glove, q_info['mask'],self.linear1)
        c_q_weighted_emb = self.AttentionLayer(a_emb_glove, q_emb_glove, q_info['mask'],self.linear2)
        c_p_weighted_emb = self.AttentionLayer(a_emb_glove, p_emb_glove, p_info['mask'],self.linear3)



        p_rnn_input,c_rnn_input,q_rnn_input=self.Concatenation(w_p,w_q,w_a,p_q_weighted_emb,c_q_weighted_emb,c_p_weighted_emb,f_tensor)        
        
        p_hiddens = self.HiddenLayer(p_rnn_input,p_info)
        q_hiddens = self.HiddenLayer(q_rnn_input,q_info)
        c_hiddens = self.HiddenLayer(c_rnn_input,a_info)
        
        
        q_hidden=self.FinalAttention(q_hiddens,q_info)
        p_hidden=self.FinalAttention([p_hiddens, q_hidden],p_info)
        c_hidden=self.FinalAttention(c_hiddens,a_info)
        
        proba=self.OutputLayer(p_hidden,q_hidden,c_hidden)
        

        return proba
    
    
    def HiddenLayer(self, rep, info):
        	#init self.stacked_rnns
    
            flag = info['flag']
            mask = info['mask']
        
            if(flag=='question'):
                q_rnn_input,q_unsort = self.pack(rep,mask)
                q_out,q_hiddens=self.question_rnn(q_rnn_input)
                q_hiddens, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(q_out)
                q_hiddens = q_hiddens.transpose(0, 1)
                q_hiddens = q_hiddens.index_select(0, q_unsort)
                return q_hiddens
            if(flag=='passage'):
                p_rnn_input,p_unsort = self.pack(rep,mask)
                p_out,p_hiddens=self.passage_rnn(p_rnn_input)
                p_hiddens, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(p_out)
                p_hiddens = p_hiddens.transpose(0, 1)
                p_hiddens = p_hiddens.index_select(0, p_unsort)
                return p_hiddens
            if(flag=='answer'):
                c_rnn_input,c_unsort = self.pack(rep,mask)
                c_out,c_hiddens=self.answer_rnn(c_rnn_input)
                c_hiddens, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(c_out)
                c_hiddens = c_hiddens.transpose(0, 1)
                c_hiddens = c_hiddens.index_select(0, c_unsort)
                return c_hiddens
                
    def Concatenation(self,w_p,w_q,w_a,p_q_weighted_emb,c_q_weighted_emb,c_p_weighted_emb,f_tensor):
        p_glove_emb, p_pos_emb, p_ner_emb, q_relation, c_relation = w_p
        q_glove_emb, q_pos_emb = w_q 
        a_glove_emb = w_a

        p_rnn_input=torch.cat([p_glove_emb, p_q_weighted_emb, p_pos_emb, p_ner_emb, f_tensor, q_relation, c_relation], dim=2)
               
        c_rnn_input=torch.cat([a_glove_emb, c_q_weighted_emb, c_p_weighted_emb], dim=2)
        q_rnn_input=torch.cat([q_glove_emb, q_pos_emb], dim=2)
        
        return p_rnn_input,c_rnn_input,q_rnn_input
     
    def InputLayer(self, item, flag=None):
    	
    	#mask, relation names, f_tensor, self.embedding

    	#init embedding funcs

    	if flag == 'passage':

    		p_glove_emb, p_pos_emb, p_ner_emb, p_mask, f_tensor, q_relation, a_relation = item

    		#print (p_pos_emb)

    		w_p = [
    		F.dropout(self.embedding(p_glove_emb),p=self.dropout,training=self.training),
    		F.dropout(self.pos_emb_func(p_pos_emb),p=self.dropout,training=self.training),
    		F.dropout(self.ner_emb_func(p_ner_emb),p=self.dropout,training=self.training),
    		F.dropout(self.rel_emb_func(q_relation),p=self.dropout,training=self.training),
    		F.dropout(self.rel_emb_func(a_relation),p=self.dropout,training=self.training),
    		]

    		p_info = {'mask': p_mask, 'f_tensor': f_tensor, 'flag': flag}

    		return (w_p, p_info)

    	if flag == 'question':

    		q_glove_emb, q_pos_emb, q_mask = item

    		w_q = [
    		F.dropout(self.embedding(q_glove_emb),p=self.dropout,training=self.training),
    		F.dropout(self.pos_emb_func(q_pos_emb),p=self.dropout,training=self.training),
    		]

    		q_info = {'mask': q_mask, 'flag': flag}

    		return (w_q, q_info)

    	if flag == 'answer':

    		a_glove_emb, a_mask = item

    		w_a = F.dropout(self.embedding(a_glove_emb),p=self.dropout,training=self.training)

    		a_info = {'mask': a_mask, 'flag': flag}

    		return (w_a, a_info)
    def AttentionLayer(self,x,y,y_mask,linear):
        
        x = linear(x.view(-1, x.size(2))).view(x.size())
        x = F.relu(x)
        y = linear(y.view(-1, y.size(2))).view(y.size())
        y = F.relu(y)
        
        scores = x.bmm(y.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        matched_seq = alpha.bmm(y)
        
        return F.dropout(matched_seq,p=self.dropout,training=self.training)
        
    def OutputLayer(self,p_hidden,q_hidden,c_hidden):
        logits = torch.sum(self.p_c_bilinear(p_hidden) * c_hidden, dim=-1)

        logits += torch.sum(self.q_c_bilinear(q_hidden) * c_hidden, dim=-1)

        proba = F.sigmoid(logits)
        
        return proba

        
    def LinearSeqAttn(self,x,x_mask,linear_layer):
        
        x_flat = x.view(-1, x.size(-1))
        scores = linear_layer(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha
    def BilinearSeqAttn(self,x,y,x_mask):
        
        Wy = self.bi_linear(y)
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(xWy)
        
        return alpha
        
    def FinalAttention(self,hiddens,info):
        flag=info['flag']
        mask=info['mask']
        if(flag=='question'):
            q_merge_weights = F.dropout(self.LinearSeqAttn(hiddens,mask,self.linear_q),p=self.dropout,training=self.training)
    
            q_hidden=q_merge_weights.unsqueeze(1).bmm(hiddens).squeeze(1)
            return q_hidden
        if(flag=='passage'):
            
            p_merge_weights = F.dropout(self.BilinearSeqAttn(hiddens[0], hiddens[1], mask),p=self.dropout,training=self.training)
            p_hidden=p_merge_weights.unsqueeze(1).bmm(hiddens[0]).squeeze(1)
            
            return p_hidden
            
        if(flag=='answer'):
            c_merge_weights = F.dropout(self.LinearSeqAttn(hiddens, mask,self.linear_c),p=self.dropout,training=self.training)
            c_hidden = c_merge_weights.unsqueeze(1).bmm(hiddens).squeeze(1)
            
            return c_hidden
        
    def pack(self,x,x_mask):
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()

        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        
        return rnn_input,idx_unsort
        
    def load_embeddings(self, words, embedding_file):

        words = {w for w in words if w in Vocab_dict}
        embedding = self.embedding.weight.data

        word_counts = {}
        with open(embedding_file) as f:
            for line in f:
                w = unicodedata.normalize('NFD', line.rstrip().split()[0].strip())
                if w in words:
                    embedding_vector = torch.Tensor([float(i) for i in line.rstrip().split()[1:]])
                    if w not in word_counts:
                        word_counts[w] = 1
                        embedding[Vocab_dict[w]].copy_(embedding_vector)
                    else:
                        word_counts[w] = word_counts[w] + 1
                        embedding[Vocab_dict[w]]+=embedding_vector

        for w, c in word_counts.items():
            embedding[Vocab_dict[w]]=embedding[Vocab_dict[w]]/c

        
def train(train_data):
    network.train()
    updates = 0
    iter_cnt, num_iter = 0, (len(train_data) + batch_size - 1) // batch_size
    for batch_input in get_batch(train_data):
        feed_input = [x for x in batch_input[:-1]]
        y = batch_input[-1]
        pred_proba = network(*feed_input)

        loss = F.binary_cross_entropy(pred_proba, y)
        network_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(network.parameters(), 10)

        fixed_embedding=network.embedding.weight.data[10:]
        network_optimizer.step()
        network.embedding.weight.data[10:] = fixed_embedding
        updates += 1
        iter_cnt += 1

        if updates % 20 == 0:
            print('Iter: %d/%d, Loss: %f' % (iter_cnt, num_iter, loss.data[0]))
    scheduler.step()
    print('LR:', scheduler.get_lr()[0])
    


def get_batch(data,batch_size=32):
    num_iter = (len(data) + batch_size - 1) // batch_size
    for i in range(num_iter):
        start_idx = i * batch_size
        batch_data = data[start_idx:(start_idx + batch_size)]
        batch_input = batchify(batch_data)

        batch_input = [Variable(x.cuda(async=True)) for x in batch_input]

        yield batch_input 
        
def evaluate(dev_data):
    Accuracy=0
    network.eval()
    Acc_vec=[]
    for batch_input in get_batch(dev_data,batch_size=2):
        feed_input = [x for x in batch_input[:-1]]
        y = batch_input[-1]
        pred_proba = network(*feed_input)
    
        predictions=pred_proba.data.cpu().numpy() 
        y1=y.data.cpu().numpy() 

        a=np.argmax([predictions[0],predictions[1]])
        b=np.argmax([y1[0],y1[1]])
        if(a==b):
            accuracy=2
        else:
            accuracy=0
        
        Accuracy+=accuracy
        Acc_vec.append(accuracy)
    print(Acc_vec)
    
    return Accuracy/len(samples_dev)



    
def batchify(batch_data):

    
    P=[]
    P_pos=[]
    P_ner=[]
    Q=[]
    Q_pos=[]
    C=[]
    F_tensor=[]
    P_q_relation=[]
    P_c_relation=[]
    Y=[] 
    P_max_len=0
    Q_max_len=0
    C_max_len=0

    for sample in batch_data:
        P.append(torch.LongTensor(sample[0]))
        P_max_len=max(P_max_len,len(sample[0]))
        P_pos.append(torch.LongTensor(sample[1]))
        P_ner.append(torch.LongTensor(sample[2]))
        Q.append(torch.LongTensor(sample[3]))
        Q_max_len=max(Q_max_len,len(sample[3]))
        Q_pos.append(torch.LongTensor(sample[4]))
        C.append(torch.LongTensor(sample[5]))
        C_max_len=max(C_max_len,len(sample[5]))
        F_tensor.append(torch.LongTensor(sample[6]))
        P_q_relation.append(torch.LongTensor(sample[7]))
        P_c_relation.append(torch.LongTensor(sample[8]))
        Y.append(sample[9])
        
    batch_size = len(P)

    p = torch.LongTensor(batch_size, P_max_len).fill_(0)
    p_mask = torch.ByteTensor(batch_size, P_max_len).fill_(1)
    for i, t in enumerate(P):
        p[i, :len(t)].copy_(t)
        p_mask[i, :len(t)].fill_(0)
        
    p_pos = torch.LongTensor(batch_size, P_max_len).fill_(0)
    for i, t in enumerate(P_pos):
        p_pos[i, :len(t)].copy_(t)
    
    p_ner = torch.LongTensor(batch_size, P_max_len).fill_(0)
    for i, t in enumerate(P_ner):
        p_ner[i, :len(t)].copy_(t)
        
    p_q_relation = torch.LongTensor(batch_size, P_max_len).fill_(0)
    for i, t in enumerate(P_q_relation):
        p_q_relation[i, :len(t)].copy_(t)
    
    p_c_relation = torch.LongTensor(batch_size, P_max_len).fill_(0)
    for i, t in enumerate(P_c_relation):
        p_c_relation[i, :len(t)].copy_(t)
        
    q = torch.LongTensor(batch_size, Q_max_len).fill_(0)
    q_mask = torch.ByteTensor(batch_size, Q_max_len).fill_(1)
    for i, t in enumerate(Q):
        q[i, :len(t)].copy_(t)
        q_mask[i, :len(t)].fill_(0)
        
    q_pos = torch.LongTensor(batch_size, Q_max_len).fill_(0)
    for i, t in enumerate(Q_pos):
        q_pos[i, :len(t)].copy_(t)
    
    c = torch.LongTensor(batch_size, C_max_len).fill_(0)
    c_mask = torch.ByteTensor(batch_size, C_max_len).fill_(1)
    for i, t in enumerate(C):
        c[i, :len(t)].copy_(t)
        c_mask[i, :len(t)].fill_(0)
        

    f_tensor = torch.FloatTensor(batch_size, P_max_len, 5).fill_(0)
    
    for i, f in enumerate(F_tensor):
        f_tensor[i, :len(f), :].copy_(f)
    
    y=torch.FloatTensor(Y)
  

    return p, p_pos, p_ner, p_mask, q, q_pos, q_mask, c, c_mask, f_tensor, p_q_relation, p_c_relation, y




network=MCQ_RNN()
network_optimizer = optim.Adamax(network.parameters(), lr=2e-3)
scheduler = lr_scheduler.MultiStepLR(network_optimizer, milestones=[10, 15], gamma=0.5)

network.cuda()


n_epochs=50
epoch=0
batch_size=32
index=0
Accuracy=0
iterations=0
done=False
best_dev_acc=0
for i in range(n_epochs):
        print('Epoch %d...' % i)
        np.random.shuffle(samples)
        cur_train_data = samples

        train(cur_train_data)
        dev_acc = evaluate(samples_dev)
        print('Dev accuracy: %f' % dev_acc)

        best_dev_acc=max(dev_acc,best_dev_acc)
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(network.state_dict(), 'network_exp.pt')



    
