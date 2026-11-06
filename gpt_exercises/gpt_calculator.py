import random
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 6
block_size = 10
n_embd = 32

n_head = 8
head_dim = 4

END = '@' # 结束符
PAD = '_' # 占位符
# ===============

# mapping
chars = "".join(str(i) for i in range(0, 10))
stoi = { s:i for i,s in enumerate(chars)}
stoi['+'] = 10
stoi['='] = 11
stoi[END] = 12 # 结束符号
stoi[PAD] = 13 # pad
itos = { i:s for s, i in stoi.items()}

vocab_size = len(itos)

# 训练集与测试集
random.seed(22)

pairlist = []
for a in range(10, 100):
    for b in range(a,100):
        pairlist.append((a,b))
random.shuffle(pairlist)
n = int(len(pairlist) * 0.9)
train_data = pairlist[:n]
val_data = pairlist[n:]

# 编码
def encode(s, is_y = False):
    out = torch.tensor([stoi[c] for c in s])
    if (is_y):
        befor_eq = False
        for i in range(out.shape[0]):
            if not befor_eq:
                befor_eq = befor_eq or out[i] == stoi['='] 
                out[i] = -1
            if out[i] == stoi[PAD]:
                out[i] = -1
            
    return out

# 获取batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 一个pair交互顺序能生成两条式子
    expressions = []
    for i in range(int(batch_size/2)):
        pair = random.choice(data)
        a = pair[0]; b= pair[1]; c = a + b
        expressions.append(f"{a}+{b}={str(c)[::-1]}@")
        expressions.append(f"{b}+{a}={str(c)[::-1]}@")
    x = torch.stack([encode(exp.ljust(block_size, PAD)) for exp in expressions])
    y = torch.stack([encode(exp[1:].ljust(block_size, PAD), True) for exp in expressions])
    return x,y

class MultiHead(nn.Module):
    """ multi head attention """
    def __init__(self, num_head, head_dim):
        super().__init__()
        self.num_head = num_head
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_embd, head_dim * num_head, bias=False)
        self.k_proj = nn.Linear(n_embd, head_dim * num_head, bias=False)
        self.v_proj = nn.Linear(n_embd, head_dim * num_head, bias=False)
        self.linear = nn.Linear(head_dim * num_head, n_embd, bias=False)

    # BTC => BHTD
    def tran(self, batch_size, time_size, input : torch.Tensor):
        out = input.view(batch_size, time_size, self.num_head, self.head_dim) # BTHD
        out = out.transpose(dim0=1, dim1=2)
        return out

    def forward(self, x):
        B, T, C = x.shape
        Q = self.tran(B, T, self.q_proj(x))
        K = self.tran(B, T, self.k_proj(x))
        V = self.tran(B, T, self.v_proj(x))

        S = Q @ K.transpose(-1,-2) * (self.head_dim **(-0.5))
        mask = torch.tril(torch.ones(T, T))
        S = S.masked_fill(mask==0, float('-inf'))
        A = F.softmax(S, dim=-1)
        O = A @ V

        O = O.transpose(1,2) # 先交换维度
        O = O.reshape(B, T, self.num_head * self.head_dim)
        out = self.linear(O)
        return out

class Feed_Froward(nn.Module):
    """ 前馈神经网络 """
    def __init__(self):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.fnn(x)

class Block(nn.Module):
    """ pre_LN + 残差网络 """
    def __init__(self):
        super().__init__()
        self.ln_mh = nn.LayerNorm(n_embd)
        self.multi_head = MultiHead(n_head, head_dim)
        self.ln_fnn = nn.LayerNorm(n_embd)
        self.fnn = Feed_Froward()

    def forward(self, x):
        x = x + self.multi_head(self.ln_mh(x))
        x = x + self.fnn(self.ln_fnn(x))
        return x

class GPTCalculateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.block = Block()
        self.layer_normal = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, target = None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T))
        x = token_emb + pos_emb
        x = self.block(x)
        x = self.layer_normal(x)
        logits = self.linear(x)

        if target is None:
            loss = None
        else:
            tmp_logits = logits.view(B*T, -1)
            target = target.view(B*T,)
            loss = F.cross_entropy(tmp_logits, target, ignore_index=-1)

        return loss, logits

model = GPTCalculateModel()
# idx, target = get_batch('train')
# loss, logits = model(idx, target)
# print(loss)

# Test1 With target
idx, target = get_batch('train')
print(f"idx shape: {idx.shape}, target shape: {target.shape}")
loss, logits = model(idx, target)
print(f"WITH TARGET loss: {loss}, logits shape: {logits.shape}")

loss.backward()
print(f" token embedding 权重 {model.token_embedding_table.weight.grad}")
print(f" 注意力 Q 投影权重 {model.block.multi_head.q_proj.weight.grad}")
print(f" FFN 第一层权重 {model.block.fnn.fnn[0].weight.grad}")
print(f" 最后输出层权重 {model.linear.weight.grad}")

# Test2 No Target
loss, logits = model(idx)
print(f"NO TARGET loss: {loss}, logits shape: {logits.shape}")
