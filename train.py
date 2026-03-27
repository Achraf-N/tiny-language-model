import os
import time
from typing import final
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataclasses import dataclass
import math
import inspect
from xml.parsers.expat import model
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # this is the linear layer that will be used to compute the query, key and value
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # this is the linear layer that will be used to project the output of the attention back to the original dimension
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)) # this is the mask that will be used to mask the future tokens
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x) # this is the linear layer that will be used to compute the query, key and value
        q, k, v = qkv.split(self.n_embd, dim=2) # this is the splitting of the query, key and value
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # this is the reshaping of the query to separate the heads
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # this is the reshaping of the key to separate the heads
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # this is the reshaping of the value to separate the heads
#------------------------------------------------------------------------------------------------------------
      
        # we compute the full atention score matrix, size (B, n_head, T, T)
        # if T is large, this can be very large it kill our GPU memory
        # Q and K are loaded from HBM into SRAM to do matrix multiplication
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # this is the computation of the attention scores
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # this is the masking of the future tokens
        #att = F.softmax(att, dim=-1) # this is the softmax of the attention scores
        #y = att @ v # this is the computation of the output of the attention

#------------------------------------------------------------------------------------------------------------

        # FlashAttention tiles the computation: it loads small blocks of Q and K into fast SRAM.
        # Computes the softmax and multiplies by V inside SRAM, without writing the intermediate att to HBM.
        # Only the final output block is written back to HBM.
        # Spped up from 85ms to 67ms"
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:,:,:T,:T], is_causal=True) # this is the computation of the attention using flash attention, it is more memory efficient and faster than the naive implementation

#------------------------------------------------------------------------------------------------------------

        y = y.transpose(1, 2).contiguous().view(B, T, C) # this is the reshaping of the output of the attention back to the original dimension
        #output projection
        y = self.c_proj(y) # this is the linear layer that will be used to project the output of the attention back to the original dimension
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # this is the expansion of the hidden layer
        self.gelu = nn.GELU(approximate='tanh') # this is the non-linearity
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # this is the projection back to the original dimension
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) #
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # this is the communication
        x = x + self.mlp(self.ln_2(x)) # this is the computation
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens 50,000 BPE merges + 256 byte tokens + 1 <|endoftext| token special token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # skeleton of a GPT 2 language model
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding table
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding table
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # block of transformer layers
            ln_f = nn.LayerNorm(config.n_embd), # layer added by gpt2 before linear
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # linear layer after the block and befor softmax

        #Weight tying = reuse the same “dictionary” for reading & writing words
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # we want to share the weights between the token embedding and the linear layer, so we set them to be the same tensor    

         #init params: Go through every submodule inside me and run _init_weights on it.
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            #we use 1/sqrt(𝑁) to control growth
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= module.NANOGPT_SCALE_INIT
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # this is the initialization of the weights of the linear layers
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # this is the initialization of the biases of the linear layers
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # this is the initialization of the weights of the embedding layers

    #before generating from the model, we need forward the input through the model.

    def forward(self, idx, targets=None):
        #input to the model is going to be our indicies "token"
        # B is the batch size, T is the time dimension, T can't be more than the block size
        # so B is a sequence and there t in this sequence and we have B independant sequences stacked up in a batch
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape T
        #pos = torch.arange(0, T, dtype=torch.long).to(device)
        position_embeddings = self.transformer.wpe(pos) # this is the position embedding (T, n_embd)
        token_embeddings = self.transformer.wte(idx) # this is the token embedding (B, T, n_embd)
        
        #there is broadcasting happening here, the position embeddings are added to the token embeddings, so the position embeddings are broadcasted across the batch dimension
        x = token_embeddings + position_embeddings # this is the input to the transformer blocks
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x) # this is the output of the transformer blocks
        x = self.transformer.ln_f(x) # this is the output of the layer norm
        # we will calculate the logits of what the next token is going to be in the sequence (B,T+1)
        logits = self.lm_head(x) # this is the output of the linear layer (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # this is the cross entropy loss between the logits and the targets
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "model_name must be one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'"
        from transformers import GPT2LMHeadModel
        print(f"Loading weights for model {model_name}...")
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25 , n_embd=1600),
        }[model_name]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
    # load the weights from the pretrained model
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # the bias is not a parameter, it is a buffer, so we need to remove it from the state dict keys
        # init huggingface model
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        # load the weights from the huggingface model
        sd_hf = model_hf.state_dict()

        #copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # the masked_bias is not a parameter, it is a buffer, so we need to remove it from the state dict keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # the bias is not a parameter, it is a buffer, so we need to remove it from the state dict keys
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a Conv1D module but we only want to use the vanilla linear layer, so we need to remove the 'c_attn.' prefix and replace it with 'attn.c_attn.' and remove the 'c_proj.' prefix and replace it with 'attn.c_proj.' and remove the 'ln_1.' prefix and replace it with 'ln_1.' and remove the 'ln_2.' prefix and replace it with 'ln_2.' and remove the 'h.' prefix and replace it with 'h.' and remove the 'wte.' prefix and replace it with 'wte.' and remove the 'wpe.' prefix and replace it with 'wpe.' and remove the 'lm_head.' prefix and replace it with 'lm_head.'
        assert len(sd_keys) == len(sd_keys_hf), "the number of parameters in the state dicts must be the same"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape, f"the shape of the parameter {k} must be the same in both state dicts"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


    # Weight decay = regularization (shrink weights) -> It prevents overfitting and can improve generalization, but it can also slow down training and reduce the final performance if it is too high, so we need to find a good balance between the two, we will use a weight decay of 0.1 in this example, which is a common value that works well for many models, but you can experiment with different values to see how it affects the training and the final performance of the model.
    # Good for large weight matrices but it break some parameters like bias and layernorm weight

    # Big matrices >=2D → prone to overfitting → decay them
    # Small control params (bias, norm) = 1D → don’t touch them
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all candidates that requires grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weighted decayed, otherewise not
        # i.e all weight tensor in matmuls + embeddings decay, all bias and layernorm weight tensors no decay
        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decay params tensors: {len(decay_params):,} | with {num_decay_params:,} parameters")
        print(f"num no decay params tensors: {len(nodecay_params):,} | with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use fused Adam if we are on CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print("using fused AdamW:", use_fused)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens)//(B*T)} batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # input (B,T) we reshape the tokens into a batch of sequences, we will use this as the input to the model
        y = (buf[1:]).view(B, T) # targets (B,T)
        #advance the position in the tensor
        self.current_position += B*T
        # if we have reached the end of the tokens, we reset the position to the beginning
        if self.current_position + B*T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y

# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
    
Total_batch_size = 524288 # 2 ** 19. 0.5M, is number of tokens
B  = 16 # micro batch size
T = 1024 # sequence length

assert Total_batch_size % (B * T) == 0, "Total batch size must be divisible by B * T"
grad_accum_steps = Total_batch_size // (B * T) # this is the number of gradient accumulation steps that we will use to achieve the desired total batch size, it is calculated by dividing the total batch size by the product of the micro batch size and the sequence length, so in this case it will be 32, which means that we will accumulate gradients for 32 steps before we update the weights of the model, this allows us to effectively use a larger batch size than what our GPU memory can handle, and it can help us stabilize the training and improve the performance of the model
print(f"Total batch size = {Total_batch_size}")
print(f"==> calculated gradient accumulation steps: {grad_accum_steps}")
train_loader = DataLoader(B=B, T=T)

#float 32 matmul precision -> time 80 ms per step before it's 130 ms per step
torch.set_float32_matmul_precision('high') #The result: 80 ms per step instead of 130 ms — a ~40% speedup.
#You’re not losing training accuracy, just using better GPU scheduling. 
# Uses optimized FP32 kernels → faster than default, still accurate


#get the logits
# random initialization of the model, random weights, not trained

model = GPT(GPTConfig(vocab_size=50304))  # we override the vocab size because we want good number divided by 12, 32, 64, 128...
# this increase speed up from 67 ms per step to 65 ms per step.
# it speed up because cuda use block tiles power of 2
model.to(device)

# this is the new feature in PyTorch 2.0 that allows us to compile the model for
#  faster inference and training, it uses a just-in-time compiler to optimize the model 
# for the specific hardware it is running on, it can give us a significant speedup during 
# training and inference, especially for large models like GPT, we will see the speedup 
# in the training loop below when we run the forward and backward passes of the model, 
# it can give us a speedup of up to 2x or more depending on the hardware and the model size, 
# it is a very powerful feature that can help us train our models faster and more efficiently,
#  we will see how to use it in the training loop below when we run the forward and 
# backward passes of the model.
# Kernel fusion = ONE optimization step
# i comment because i need install triton in windows
#model = torch.compile(model)

#---------------------------------------------------------------------------------------------------------
# dynamic α:
#early → large steps (learn fast)
#late → small steps (refine)
# learning rate scheduler
max_lr = 6e-4
min_lr = 0.1*max_lr
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    #if it >learn_decay_iters return min_lr
    if it > max_steps:
        return min_lr
    # in between use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, "decay_ratio must be between 0 and 1"
    coef = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coef * (max_lr - min_lr)


#---------------------------------------------------------------------------------------------------------

#logits, loss = model(x)
#print(logits.shape) 
# right now we intitialize the vocab size to 50257 but it's uniformly random probabilities.
#Vocabulary elements have uniform probability
# we need training we use adam optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8) # this is the AdamW optimizer that we will use to train the model, it is a variant of the Adam optimizer that decouples the weight decay from the learning rate, which can give us better performance and stability during training
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device) 

# we use 50 steps of training
# 50 steps =
#    50 batches (maybe random)
#    50 forward passes
#    50 backward passes
#    50 updates
for i in range(50):
    t0 = time.time()
    # we start by zeroing the gradients of the optimizer, then we forward the input through the model to get the logits and the loss, then we backpropagate the loss to get the gradients, and then we step the optimizer to update the weights of the model, and then we print the loss for this step
    optimizer.zero_grad()
    for k in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            # now the activation tensor is bf16 not everything change the weights stay torch float 32
            logits, loss = model(x, y)
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # this is the gradient clipping that we will use to prevent the exploding gradients problem, it clips the gradients to a maximum norm of 1.0, which can help us stabilize the training and prevent the gradients from becoming too large
    optimizer.step()
    torch.cuda.synchronize() # we synchronize the GPU to make sure that all the operations are finished before we measure the time
    t1 = time.time()
    dt = (t1 - t0) # time difference in seconds
    token_processed = train_loader.B * train_loader.T
    tokens_per_sec = token_processed / dt
    print(f"step {i+1}: loss {loss.item():.4f}, | lr: {get_lr(i):.2e} | time {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | grad norm: {norm:.2f}")

    #print(f"step {i+1}: loss {loss.item()}")

import sys; sys.exit(0) # we exit here because we just want to test if the model can be loaded without crashing, we will do the generation in the next step
num_return_sequences = 5
max_length = 30
#these layeres are identical in both training and evaluation, because we doesn't use dropout or any other regularization techniques that behave differently during training and evaluation, so we can just use the same model for both training and evaluation
# we initialize the model from the pretrained weights
#model is guessing randomly ->every token equally likely
model = GPT.from_pretrained('gpt2')
# we set the model to evaluation mode, so no training we just use the model
model.eval()
# we moving the model to the GPU, if available, so we can use the GPU for inference
model.to(device)
#print("didn't crash so it worked!")

#prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
token = enc.encode("Hello, I'm a language model,") # there are 8 tokens in this string, and the model will predict the next token after this string
tokens = torch.tensor(token, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8) we add a batch dimension and move the tokens to the GPU
x = tokens.to(device) # (5, 8)

#generate right now x is (B, T) where B=5 and T=8
#set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# for each iteration in these loop we are going to be adding a column of indices at each one of these rows
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size) we only care about the last time step, because we want to predict the next token
        probs = F.softmax(logits, dim=-1) # (B, vocab_size) we convert the logits to probabilities
        # do top k sampling of 50
        #topk_probs here become (5,50) topk_indices becomes (5,50)
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1) # (B, k) we get the top k probabilities and their corresponding indices
        #select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1) we sample from the top k probabilities
        # gather the corresponding token indices for the sampled probabilities
        x_col = torch.gather(topk_indices, dim=-1, index=ix) # (B,1)
        # append  to the sequence
        x = torch.cat((x, x_col), dim=1) # (B, T+1) we append the next token to the sequence


#print the generated sequences
for i in range(num_return_sequences):
    generated_tokens = x[i,:max_length].tolist() # (T,)
    decoded = enc.decode(generated_tokens) # we decode the generated tokens back to text
    print(">",decoded)















