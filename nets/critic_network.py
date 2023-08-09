from torch import nn
from nets.graph_layers import  MultiHeadAttentionLayerforCritic, ValueDecoder

class Critic(nn.Module):

    def __init__(self,
             embedding_dim,
             hidden_dim,
             n_heads,
             n_layers,
             normalization,
             with_regular,
             with_bonus
             ):
        
        super(Critic, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.normalization = normalization
        self.with_regular = with_regular
        self.with_bonus = with_bonus
        
        self.encoder = MultiHeadAttentionLayerforCritic(self.n_heads, 
                                self.embedding_dim,
                                self.hidden_dim, 
                                self.normalization)
            
        self.value_head = ValueDecoder(input_dim = self.embedding_dim,
                                       embed_dim = self.embedding_dim,
                                       with_regular = self.with_regular,
                                       with_bonus = self.with_bonus)
        
        print('# params in Critic', self.get_parameter_number())
        
    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
        
    def forward(self, input, cost, context2):
        # pass through encoder
        h_em = self.encoder(input.detach())

        # pass through value_head, get estimated value
        baseline_value = self.value_head(h_em, cost, context2)
        
        # return baseline_value.detach().squeeze(), baseline_value.squeeze()
        return baseline_value.detach(), baseline_value
        
