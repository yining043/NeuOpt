import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch import nn
import math

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q):
        
        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)   
       
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out

class SynthAtt(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            number_aspect,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
    ):
        super(SynthAtt, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
         
        self.score_aggr = nn.Sequential(
                        nn.Linear(self.n_heads * number_aspect, self.n_heads * 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.n_heads * 2, self.n_heads))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()
        
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_em, route_attn):
        
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h_em.size()
        hflat = h_em.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, bastch_size, n_query, graph_size)
        compatibility = torch.cat((torch.matmul(Q, K.transpose(2, 3)), route_attn), 0)
       
        mixed_raw = compatibility.permute(1,2,3,0)
        attn = self.score_aggr(mixed_raw).permute(3,0,1,2)
        heads = torch.matmul(F.softmax(attn, dim=-1), V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        return out, route_attn

class MultiHeadPosCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadPosCompat, self).__init__()
    
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h):
        
        batch_size, graph_size, input_dim = h.size()
        posflat = h.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(posflat, self.W_query).view(shp)  
        K = torch.matmul(posflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        return torch.matmul(Q, K.transpose(2, 3))

class MLP(torch.nn.Module):
    def __init__(self,
                input_dim = 128,
                feed_forward_dim = 64,
                embedding_dim = 64,
                output_dim = 1,
                p_dropout = 0.001
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace = True)
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result

class ValueDecoder(nn.Module): # seperate critics
    def __init__(
            self,
            embed_dim,
            input_dim,
            with_regular,
            with_bonus
    ):
        super(ValueDecoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.with_regular = with_regular
        self.with_bonus = with_bonus
        
        self.project_graph = nn.Linear(self.input_dim, self.embed_dim // 2)
        self.project_node = nn.Linear(self.input_dim, self.embed_dim // 2) 
        
        self.reward_MLP = MLP(input_dim + 1, embed_dim)
        if self.with_regular:
            self.regulation_MLP = MLP(input_dim + 10, embed_dim)
        if self.with_bonus:
            self.bonus_MLP = MLP(input_dim + 1, embed_dim)

    def forward(self, h_em, cost, context2): 
        # get embed feature
        mean_pooling = h_em.mean(1)     # mean Pooling
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        
        # pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature)
        
        fusion_feature_reward = torch.cat((fusion.mean(1),
                                           fusion.max(1)[0],
                                           cost[:,1:2].to(h_em.device),
                                           ), -1)
        value_reward = self.reward_MLP(fusion_feature_reward)
        
        if self.with_regular:
            fusion_feature_reg = torch.cat((fusion_feature_reward,
                                            context2), -1)
            value_regulation = self.regulation_MLP(fusion_feature_reg)
        else:
            value_regulation = value_reward * 0.0
            
        if self.with_bonus:
            fusion_feature_bonus = torch.cat((fusion.mean(1),
                                              fusion.max(1)[0],
                                              cost[:,2:].to(h_em.device),
                                              ), -1)
            value_bonus = self.bonus_MLP(fusion_feature_bonus)
        else:
            value_bonus = value_reward * 0.0

        value = torch.cat((value_reward.unsqueeze(-1),
                           value_regulation.unsqueeze(-1),
                           value_bonus.unsqueeze(-1)), -1)
            
        return value # bs, 3(reward, regulation, bonus)

class kopt_Decoder(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            v_range = 6,
            k = 5,
            with_RNN = True,
            with_feature3 = True,
            simpleMDP = False
    ):
        super(kopt_Decoder, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.key_dim = input_dim
        self.range = v_range
        self.with_RNN = with_RNN
        self.with_feature3 = with_feature3
        self.simpleMDP = simpleMDP
        print('simpleMDP: ', self.simpleMDP)
        assert simpleMDP
        
        self.linear_K1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_K2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_K3 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_K4 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        self.linear_Q1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_Q2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_Q3 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.linear_Q4 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        if self.with_feature3:
            self.meta_linear = nn.Sequential(
                            nn.Linear(9, 8),
                            nn.ReLU(inplace=True),
                            nn.Linear(8, self.embed_dim * 2))                
        else:
            self.linear_V1 = nn.Parameter(torch.Tensor(self.embed_dim))
            self.linear_V2 = nn.Parameter(torch.Tensor(self.embed_dim))
        
        if self.with_RNN:
            self.init_hidden_W = nn.Linear(self.embed_dim, self.embed_dim)
            self.init_query_learnable = nn.Parameter(torch.Tensor(self.embed_dim))
            self.rnn1 = nn.GRUCell(self.embed_dim, self.embed_dim)
            self.rnn2 = nn.GRUCell(self.embed_dim, self.embed_dim)
        else:
            self.init_query_learnable = nn.Parameter(torch.Tensor(self.embed_dim))
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            
    def forward(self, problem, h, rec, context2, visited_time, last_action, fixed_action = None, require_entropy = False):    
         
          bs, gs, _, ll, action, entropys = *h.size(), 0.0, None, []
          action_index = torch.zeros(bs, problem.k_max, dtype=torch.long).to(rec.device)
          k_action_left = torch.zeros(bs, problem.k_max + 1, dtype=torch.long).to(rec.device)
          k_action_right = torch.zeros(bs, problem.k_max, dtype=torch.long).to(rec.device)
          next_of_last_action = torch.zeros_like(rec[:,:1], dtype=torch.long).to(rec.device) - 1
          mask = torch.zeros_like(rec, dtype=torch.bool).to(rec.device)
          stopped = torch.ones(bs, dtype=torch.bool).to(rec.device)
          h_mean = h.mean(1)
          init_query = self.init_query_learnable.repeat(bs,1)
          input_q1 = input_q2 = init_query.clone()
          
          if self.with_RNN:
              init_hidden = self.init_hidden_W(h_mean)
              q1 = q2 = init_hidden.clone()
              
          if self.with_feature3:
              decoder_condition = context2
              linear_V = self.meta_linear(decoder_condition)
              linear_V1 = linear_V[:,:self.embed_dim]
              linear_V2 = linear_V[:,self.embed_dim:]
          else:
              linear_V1 = self.linear_V1.view(1, -1).expand(bs, -1)
              linear_V2 = self.linear_V2.view(1, -1).expand(bs, -1)
          
          for i in range(problem.k_max):
                      
              # GRUs
              if self.with_RNN:
                  q1 = self.rnn1(input_q1, q1)
                  q2 = self.rnn2(input_q2, q2)
              else:
                  q1 = input_q1
                  q2 = input_q2

              # Dual-Stream Attention
              result = (linear_V1.unsqueeze(1) * torch.tanh(self.linear_K1(h) + 
                                                    self.linear_Q1(q1).unsqueeze(1) +
                                                    self.linear_K3(h) * self.linear_Q3(q1).unsqueeze(1)
                                                    )).sum(-1)      # \mu stream       
              result+= (linear_V2.unsqueeze(1) * torch.tanh(self.linear_K2(h) + 
                                                    self.linear_Q2(q2).unsqueeze(1) + 
                                                    self.linear_K4(h) * self.linear_Q4(q2).unsqueeze(1)
                                                    )).sum(-1)      # \lambda stream 
              
              # Calc probs
              logits = torch.tanh(result) * self.range
              # assert (~mask).any(-1).all(), (i, (~mask).any(-1))
              logits[mask.clone()] = -1e30
              if i == 0 and isinstance(last_action, torch.Tensor):
                  logits.scatter_(1, last_action[:,:1], -1e30)
              probs = F.softmax(logits, dim = -1)
              
              # Sample action for a_i
              if fixed_action is None:
                  action = probs.multinomial(1)
                  value_max, action_max = probs.max(-1,True) ### fix bug of pytorch
                  action = torch.where(1-value_max.view(-1,1)<1e-5, action_max.view(-1,1), action) ### fix bug of pytorch
              else:
                  action = fixed_action[:,i:i+1]
                  
              if self.simpleMDP and i > 0:
                  action = torch.where(stopped.unsqueeze(-1), action_index[:,:1], action)
                  
              # Record log_likelihood and Entropy
              if self.training:
                  loss_now = F.log_softmax(logits, dim = -1).gather(-1, action).squeeze()
                  if self.simpleMDP and i > 0:
                      ll = ll + torch.where(stopped, loss_now * 0, loss_now)
                  else:
                      ll = ll + loss_now
                  if require_entropy:
                      dist = Categorical(probs, validate_args=False)
                      entropys.append(dist.entropy())                  
              
              # Store and Process actions
              next_of_new_action = rec.gather(1, action)
              action_index[:,i] = action.squeeze().clone()
              k_action_left[stopped,i] = action[stopped].squeeze().clone()
              k_action_right[~stopped, i-1] = action[~stopped].squeeze().clone()
              k_action_left[:,i+1] = next_of_new_action.squeeze().clone()
              
              # Prepare next RNN input
              input_q1 = h.gather(1, action.view(bs,1,1).expand(bs,1,self.embed_dim)).squeeze(1)
              input_q2 = torch.where(stopped.view(bs,1).expand(bs,self.embed_dim), input_q1.clone(),
                            h.gather(1, (next_of_last_action % gs).view(bs,1,1).expand(bs,1,self.embed_dim)).squeeze(1))
              
              # Process if k-opt close
              # assert (input_q1[stopped] == input_q2[stopped]).all()
              if self.simpleMDP and i > 0:
                  stopped = stopped | (action == next_of_last_action).squeeze()
              else:
                  stopped = (action == next_of_last_action).squeeze()
              # assert (input_q1[stopped] == input_q2[stopped]).all()          
              
              k_action_left[stopped, i] = k_action_left[stopped, i-1]
              k_action_right[stopped, i] = k_action_right[stopped, i-1]
              
              # Calc next basic masks
              if i == 0: 
                  visited_time_tag = (visited_time - visited_time.gather(1, action)) % gs
              mask &= False
              mask[(visited_time_tag <= visited_time_tag.gather(1, action))] = True
              if i == 0:
                  mask[visited_time_tag > (gs - 2) ] = True
              mask[stopped, action[stopped].squeeze()] = False # allow next k-opt starts immediately
              # if True:#i == problem.k_max - 2: # allow special case: close k-opt at the first selected node
              index_allow_first_node = (~stopped) & (next_of_new_action.squeeze() == action_index[:,0])
              mask[index_allow_first_node, action_index[index_allow_first_node,0]] = False
              
              # Move to next 
              next_of_last_action = next_of_new_action
              next_of_last_action[stopped] = -1
              
          # Form final action
          k_action_right[~stopped,-1] = k_action_left[~stopped,-1].clone()
          k_action_left = k_action_left[:, :problem.k_max]
          action_all = torch.cat((action_index, k_action_left, k_action_right), -1)
          
          return action_all, ll, torch.stack(entropys).mean(0) if require_entropy and self.training else None

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)
            
    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1,2)).view(-1,1,1)) / torch.sqrt(input.var((1,2)).view(-1,1,1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            number_aspect = 2,
            normalization='layer',
    ):
        super(MultiHeadEncoder, self).__init__()
        
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                        number_aspect = number_aspect
                )
        
        self.FFandNorm_sublayer = FFandNormsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        return self.FFandNorm_sublayer(out1), out2
    
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            number_aspect = 2,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()
        
        self.MHA = SynthAtt(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    number_aspect = number_aspect
                )
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)
        
        # Normalization
        return self.Norm(out1 + input1), out2
   
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()
        
        self.FF = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden, bias = False),
                    nn.ReLU(inplace = True),
                    nn.Linear(feed_forward_hidden, embed_dim, bias = False)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias = False)
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input):
    
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)

class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            node_dim,
            embedding_dim,
            seq_length,
        ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Sequential(
                nn.Linear(node_dim, embedding_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim // 2, embedding_dim))
        self.pattern = self.cyclic_position_encoding_pattern(seq_length, embedding_dim)
            
        self.init_parameters()
        
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
            
    def basesin(self, x, T, fai = 0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    def basecos(self, x, T, fai = 0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    def cyclic_position_encoding_pattern(self, n_position, emb_dim, mean_pooling = True):
        
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype = 'int')
        x = np.zeros((n_position, emb_dim))
         
        for i in range(emb_dim):
            Td = Td_set[i //3 * 3 + 1] if  (i //3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else  2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 ==1:
                x[:,i] = self.basecos(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
            else:
                x[:,i] = self.basesin(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
                
        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)

        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else[-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1,1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)
        #### ---- 
        
        return pattern
    

    def position_encoding(self, base, embedding_dim, order_vector):
         batch_size, seq_length = order_vector.size()
         
         # expand for every batch
         position_enc = base.expand(batch_size, *base.size()).clone().to(order_vector.device)
         
         # get index according to the solutions
         index = order_vector.unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)
         
         # return 
         return torch.gather(position_enc, 1, index)

        
    def forward(self, x, solutions, visited_time):
        
        bs, gs = solutions.size()
        x_embedding = self.embedder(x)   
        pos_enc = self.position_encoding(self.pattern, self.embedding_dim, visited_time)  
          
        return x_embedding, pos_enc

class MultiHeadAttentionLayerforCritic(nn.Sequential):
    
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                    MultiHeadAttention(
                        n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim
                    )                
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(inplace = True),
                        nn.Linear(feed_forward_hidden, embed_dim,)
                    ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )