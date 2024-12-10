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
5. Flash Attention - 1.4k, 2000 (time is faster but memory is same)
6. changed vocab size to 50304 - 1.2k, 2000 (little faster)
7. gradient accumulation - step 9: loss 7.5856 | dt 6795.20ms | tokens/sec 1205.56 | norm 2.46 | lr 6.00e-04 | tokens 8,192
"""
"""
Session3: controlling lr and weight decay
num decayed parameter tensors: 50, with 124,354,560 parameters   
num non-decayed parameter tensors: 98, with 121,344 parameters
"""
"""
Clarifications : 0.5M batch size in our code is 0.5M tokens in a batch
so 0.5M/T 
I cant use 0.5M batch size because of memory constraints
so there is a technique called gradient accumulation which will run serially but will accumulate the gradients and then do the step
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time
import inspect

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
        
        """attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        """
        # FlashAttention is a method to compute attention with lower memory usage and faster speed.
        # It uses a combination of tiling and recomputation to reduce the memory footprint.
        # This allows for larger batch sizes and sequence lengths during training and inference.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
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

    # decay - wirghts participate in matrix mul, embeddings , no_decay - bias, 1D tensors, layernorm 
    #so without applying this fucntion we are applying weight decay to all the parameters which will cause problem because we forcing to distriubute the weight decay to all the parameters
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # if master_process:
        #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        # fused is a custom kernel that fuses the elementwise operation and the step into a single kernel basically it does the same thing but in one go
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # if master_process:
        #     print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
    
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
model = GPT(GPTConfig(vocab_size=50304)) #nice number not ugly
model = model.to('cuda')

# model = torch.compile(model) #makes it faster doesnt use python interpretar like trying to intrepret one by one but compile has context of full code
#in technical terms instead by mulitple read/reads from gpu to cpu and back to gpu it does in one go

#cosine decay given in gpt3 paper
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 10
def get_lr(it):
    # 1 . Linear warmup for warmup iterations
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2 . if it > lr_decay_iters, return min_lr
    if it >= max_steps:
        return min_lr
    # 3 . in between, do cosine decay to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
# train_loader = DataLoaderLite(4, 32)
# train_loader = DataLoaderLite(16//2, 1024//2)
# gradient accumulation
# total_batch_size = 524288 # 0.5M tokens in a batch
total_batch_size = 8*1024 # 8k tokens in a batch for my pc
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B*T) == 0, "Make sure total batch size is divisible by B*T"
grad_accum_steps = total_batch_size // (B*T)
print(f"gradient accumulation steps: {grad_accum_steps}")
print(f"Calculating {grad_accum_steps} times before doing a step")

train_loader = DataLoaderLite(B, T)

torch.set_float32_matmul_precision('high')
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) #session2
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) #gpt3 paper
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='cuda')
for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to('cuda'), y.to('cuda')
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): #parameters are in float32 but activations are in bfloat16 Check what converts to bfloat and what remains same
            logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # detaching the graph so that i use only value and not store the graph
        loss.backward() # remeber the issue loss reduction mean
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip gradients to avoid exploding gradients
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() #waiting for gpu to finish
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {i}: loss {loss.item():.4f} | dt {dt:.2f}ms | tokens/sec {tokens_per_sec:.2f} | norm {norm:.2f} | lr {lr:.2e} | tokens {tokens_processed:,}")
    


