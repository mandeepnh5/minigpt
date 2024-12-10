"""
GPU notes:
reducing from 32 to 16 bit will increase by 8x and with bfloat 16 will increase by 16x(Use Tensor core or normal core
do read about bfloat and float it had a nice image showing cutting down precision bits
and generally INT 8 used in inference(production) and float 16 in training
"""
"""
Time history: RTX3050
1. 10k, 400
2. torch.set_float32_matmul_precision('high') - 6k, 700 (~2x)
3. with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 4.5k, 900 (~2.3x)
4. torch.compile(model) - not availabble on windows it could have increased by 10x
5.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #not really a bias more of a mask but follwing the original code
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))    
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        #some bug see video 1:02:00
        #weight sharing scheme
        #because of this bug 30% of the weights are not shared but after doing its wroking better--> 768 * 50257 = 40M which is 30% of 124M
        #the issue was the token embedding below of arch (below the box in paper) has the same size as the lm_head which is top after box so pytorch thinks its pointing to the same shape and identical tensor but without writing the below code we r not keeping it same so make it same and in paper they mentioned they want it be to identical
        self.transformer.wte.weight = self.lm_head.weight 
        
        #initialize params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # remeber from previous playing code why its divided because the std is increasing but we bring to near to 1 and 2* comes from self attn amd mlp see dunction of forward in block class    
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) #0.02 because roughly d(model) size --> 1/sqrt(dmodelsize) ~ 0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        # idx of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward model size {T} > {self.config.block_size}"
        # forward the tokens and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #Shape (T)
        pos_emb = self.transformer.wpe(pos) #Positional embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #Token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the network
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm
        x = self.transformer.ln_f(x)
        # forward the language model head
        logits = self.lm_head(x) #Shape (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        config = GPTConfig()
        model = GPT(config)
        sd = model.state_dict()
        # print(sd)
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('attn.bias') ]
        return model
    
    
num_return_sequences = 5 # number of sentences to generate
max_length = 30 # maximum length of the sentence


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        #at init load toekns
        with open('dataset.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Total tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")
        
        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B*T
        if self.current_position+ B*T+1 > len(self.tokens):
            self.current_position = 0
        return x, y
    

# gpt logits
model = GPT(GPTConfig())
model = model.to('cuda')
model = torch.compile(model) #makes it faster doesnt use python interpretar like trying to intrepret one by one but compile has context of full code
#in technical terms instead by mulitple read/reads from gpu to cpu and back to gpu it does in one go

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


# train_loader = DataLoaderLite(4, 32)
train_loader = DataLoaderLite(16//2, 1024//2)
torch.set_float32_matmul_precision('high')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(4):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to('cuda'), y.to('cuda')
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16): #parameters are in float32 but activations are in bfloat16 Check what converts to bfloat and what remains same
        logits, loss = model(x, y)
        # import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() #waiting for gpu to finish
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {i}: loss {loss.item():.4f}, dt {dt:.2f}ms, tokens/sec {tokens_per_sec:.2f}")
    
import sys; sys.exit(0)

# generate! right now x is (B, T) where B = 5 , T = 8
# set the seed tp 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) #Shape (B, T, vocab_size)
        # take the logits at the last position and divide by temperature
        logits = logits[:, -1, :] #Shape (B, vocab_size)
        # get the prob
        probs = F.softmax(logits, dim=-1)
        # do top_k samplingof 50(hugging face pipeline default)
        # topk_probs, here becomes (5, 50), topk_indices becomes (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # sample from the topk indices
        ix = torch.multinomial(topk_probs, num_samples=1) #Shape (5, 1)
        #gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) #Shape (5, 1)
        #concatenate to the running sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated sentences
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(decoded)


