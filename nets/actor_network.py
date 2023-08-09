import torch
from torch import nn
from nets.graph_layers import MultiHeadEncoder, EmbeddingNet, MultiHeadPosCompat, kopt_Decoder

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class Actor(nn.Module):

    def __init__(self,
                 problem,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 v_range,
                 seq_length,
                 k,
                 with_RNN,
                 with_feature1,
                 with_feature3,
                 with_simpleMDP
                 ):
        super(Actor, self).__init__()

        problem_name = problem.NAME
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length                
        self.k = k
        self.with_RNN = with_RNN
        self.with_feature1 = with_feature1
        self.with_feature3 = with_feature3
        self.with_simpleMDP = with_simpleMDP
        
        if problem_name == 'tsp':
            self.node_dim = 2
        elif problem_name == 'cvrp':
            self.node_dim = 8 if self.with_feature1 else 6
        else:
            raise NotImplementedError()
            
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.seq_length)
        
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim,
                                number_aspect = 2,
                                normalization = self.normalization
                                )
            for _ in range(self.n_layers)))

        self.pos_encoder = MultiHeadPosCompat(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim, 
                                )
        
        self.decoder = kopt_Decoder(self.n_heads_actor, 
                                    input_dim = self.embedding_dim, 
                                    embed_dim = self.embedding_dim,
                                    v_range = self.range,
                                    k = self.k,
                                    with_RNN = self.with_RNN,
                                    with_feature3 = self.with_feature3,
                                    simpleMDP = self.with_simpleMDP
                                    )

        print('# params in Actor', self.get_parameter_number())
        
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, problem, batch, x_in, solution, context, context2,last_action, fixed_action = None, require_entropy = False, to_critic = False, only_critic  = False):
        # the embedded input x
        bs, gs, in_d = x_in.size()
        
        if problem.NAME == 'cvrp':
            
            visited_time, to_actor = problem.get_dynamic_feature(solution, batch, context)
            if self.with_feature1:
                x_in = torch.cat((x_in, to_actor), -1)
            else:
                x_in = torch.cat((x_in, to_actor[:,:,:-2]), -1)
            del context, to_actor

        elif problem.NAME == 'tsp':
            visited_time = problem.get_order(solution, return_solution = False)
        else: 
            raise NotImplementedError()
            
        h_embed, h_pos = self.embedder(x_in, solution, visited_time)
        aux_scores = self.pos_encoder(h_pos)
        
        h_em_final, _ = self.encoder(h_embed, aux_scores)
        
        if only_critic:
            return (h_em_final)
        
        action, log_ll, entropy = self.decoder(problem,
                                               h_em_final,
                                               solution,
                                               context2,
                                               visited_time,
                                               last_action,
                                               fixed_action = fixed_action,
                                               require_entropy = require_entropy)
        
        # assert (visited_time == visited_time_clone).all()
        if require_entropy:
            return action, log_ll, (h_em_final) if to_critic else None, entropy
        else:
            return action, log_ll, (h_em_final) if to_critic else None