#!/usr/bin/env python
# coding: utf-8

# # **Assignment 2: Building a Simple Language Model**
# 
# In this assignment, you will build and extend a tiny neural language model step by step. Starting from a single-vector bag-of-words baseline, you will gradually add positional information, non-linear transformations, a simplified form of attention, MLP layers, residual connections, and gating mechanisms to see how each architectural choice changes the model's ability to predict the next token.
# 
# From this assignment, you will know how to construct the core modules of a language model, train it on data, and understand how architectural design choices affect model performance.
# 
# **Key Components You Will Implement**:
# 
# **Cross-Entropy Loss**: Measures next-token prediction accuracy.\
# **Training Loop**: Forward pass, loss computation, and gradient update.\
# **Positional Embeddings**: Encodes token order to enable sequence modeling.\
# **Simplified Attention**: Replaces the equal-weight average with a similarity-weighted one.\
# **Feed-Forward Network (MLP)**: Learning a function of previous token that is more useful than a simple average\
# **Residual Connections**: Stabilizes training and enhances gradient flow.\
# **Gating Mechanism**: Regulates information flow across layers.
# 
# You will write code inside the `TODO` blocks and answer short conceptual questions.
# 
# At the end you will write a paragraph or two discussing your findings
# 
# We will be using a logging framework called TensorBoard to record metrics during training in order to visualize and compare metrics across different architectures.
# 
# A GPU is recommended but not required. You can select the  T4 GPU through Colab by going to Runtime > Change Runtime Type > T4 GPU

# # **Imports and Dependencies**

# In[ ]:


!pip install pygtrie datasets

# In[ ]:


import itertools
import json
from collections import Counter
from typing import Dict, List
import datasets
import numpy as np
import pygtrie
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

# # **Tokenizer**
# 
# Here is the same byte-pairing tokenizer that you implemented in A0, except it has been trained with a vocab size of 65536 on linux text data and a special token has been added to identify the end of a document. Try it on a couple example sentences.
# 

# In[ ]:


class Tokenizer:
    def __init__(self, vocab_path=None, vocab=None, max_token_length=30, partial_trie=None, special_tokens = None):
        """
        special_tokens: optional dict like {'eos_token': '<eos>', 'pad_token': '<pad>'}
        These will be assigned fresh IDs at the end of the vocab and stored separately.
        """
        print(f"\nInitializing Tokenizer:")
        print(f"  - vocab_path: {vocab_path}")
        print(f"  - vocab size: {len(vocab) if vocab else 'None'}")
        print(f"  - max_token_length: {max_token_length}")
        self.vocab_path = vocab_path
        self.id_to_tok = []
        self.trie = pygtrie.Trie()
        self.trie = self.trie if partial_trie is None else partial_trie
        self.max_token_length = max_token_length

        #adding a special eos token to seperate between rows
        self.special_tokens = {}
        self.special_id_to_name = {}

        if vocab_path or vocab:
            self.update_trie(vocab)

        if special_tokens:
            self.add_special_tokens(special_tokens)
    @property
    def vocab_size(self):
        return len(self.id_to_tok) + len(self.special_tokens)

    @property
    def eos_token(self):
      return self.special_tokens.get('eos_token', (None,None))[0]

    @property
    def eos_token_id(self):
      return self.special_tokens.get('eos_token', (None,None))[1]

    def add_special_tokens(self, mapping):
        """
        mapping: {'eos_token': '<eos>', 'pad_token': '<pad>', ...}
        Assigns new IDs after the base vocab (not added to trie).
        """
        base = len(self.id_to_tok)
        for i, (name, text_token) in enumerate(mapping.items()):
            tok_id = base + len(self.special_tokens)
            self.special_tokens[name] = (text_token, tok_id)
            self.special_id_to_name[tok_id] = name
        print(f"Added special tokens: { {k:v[0] for k,v in self.special_tokens.items()} }")

    def update_trie(self, new_vocab=None):
        print("\nUpdating trie...")
        if new_vocab is None and self.vocab_path:
            print(f"Loading from vocab file: {self.vocab_path}")
            with open(self.vocab_path, 'r') as f:
                for i, line in enumerate(f):
                    token = tuple(json.loads(line)[0])
                    self.id_to_tok.append(token)
                    self.trie[token] = i
        elif new_vocab:
            for token in new_vocab:
                print(token)
                self.id_to_tok.append(token)
                self.trie[token] = len(self.trie)

    def encode(self, text):
        return self._tokenize(text, return_ids=True)

    def decode(self, token_ids):
      out_bytes = bytearray()
      for tid in token_ids:
          if 0 <= tid < len(self.id_to_tok):
              out_bytes.extend(self.id_to_tok[tid])
          # ids outside the range are simply ignored
      return out_bytes.decode('utf-8', errors='ignore')


    def _tokenize(self, text, return_ids=False):
        if isinstance(text, str):
            text = text.encode('utf-8', errors='ignore')
        tokens = []
        while text:
            prefix = text[:self.max_token_length]
            longest = self.trie.longest_prefix(prefix)
            if not longest:
                print(f"Warning: No token found for prefix: {prefix[:10]}...")
                break
            tokens.append(longest.value if return_ids else longest.key)
            text = text[len(longest.key):]
        return tokens

# In[ ]:


import datasets
dataset = datasets.load_dataset('coms4705-hewitt/linuxlike-tokenized', 'default', streaming=True)['train']
with open('vocab-65k-fw-byte.txt', 'w') as f:
    for line in dataset:
        f.write(line['text'] + '\n')
tok = Tokenizer(vocab_path='vocab-65k-fw-byte.txt', special_tokens={'eos_token': '<eos>'})
print(tok.eos_token, tok.eos_token_id)
print(tok.vocab_size)

# # **Single Vector Model**
# 
# In class you've learned some rudimentary ways in which we can build representations of text. Now, we can use it for next token prediction. Here is a very simplified neural language model, on top of which you will be building your own model. It is equivalent to the model presented in Lecture 1: Text Representation and Language Modeling.

# In[ ]:


class SingleVectorWordModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Initialize the model with a single vector per word.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super().__init__()

        self.embeddings1 = nn.Embedding(vocab_size, embedding_dim)

        # Initialize weights to small values
        nn.init.uniform_(self.embeddings1.weight, -0.001, 0.001)

    def encode_context(self, context_indices):
        emb1 = self.embeddings1(context_indices)  # (bs, seq, d)
        return torch.sum(emb1, dim=1) / context_indices.size(1) # (bs, d)

    def score(self, h):
        return torch.einsum('vd,bd->bv', self.embeddings1.weight, h)

    def forward(self, context_indices):
        """
        Forward pass of the model.

        Args:
            context_indices: Tensor of shape (batch_size, 3) containing indices of context words

        Returns:
            logits: Tensor of shape (batch_size, vocab_size) containing prediction logits
        """
        h = self.encode_context(context_indices)
        logits = self.score(h)
        return logits

# # **Training**
# 
# Now that we have defined our model, we can start training!
# **Tasks**
# 1. Question: What does the find_similar_words method and print_similar_words method do?
# 2. Implement the function `cross_entropy_loss` using
#    `F.log_softmax` and tensor indexing (`torch.gather` or advanced indexing).
# 3. Fill in the missing steps in `train_model`:
#    * move tensors to the correct device
#    * zero gradients
#    * forward pass → logits
#    * compute loss
#    * backward pass
#    * optimizer & scheduler steps
# 4. Look at the visualization in Tensorboard and play with the interface.
# 5. Question: What do you notice about the loss of the first epoch across multiple training runs. Does this surprise you? Why or why not?

# In[ ]:


def find_similar_words(model, tokenizer, query_words: List[str], k: int = 5) -> Dict[str, List[str]]:
    embeddings_list = {'emb1': model.embeddings1.weight}
    if hasattr(model, 'embeddings2'):
        embeddings_list['emb2'] = model.embeddings2.weight
    for name, embeddings in embeddings_list.items():
        embeddings = F.normalize(embeddings, p=2, dim=1)
        results = {}
        for query in query_words:
            query_id = tokenizer.encode(query)[0]
            query_embed = embeddings[query_id]
            similarities = torch.matmul(embeddings, query_embed)
            top_k = torch.topk(similarities, k=k+1)
            similar_words = [tokenizer.decode([idx.item()]) for idx in top_k.indices[1:]]
            results[query] = similar_words
        yield results

def print_similar_words(model, tokenizer, query_words: List[str], k: int = 5):
    itr = find_similar_words(model, tokenizer, query_words, k)
    for results in itr:
        print("\nMost similar words:")
        for query, similar in results.items():
            print(f"\n'{query}':")
            for word in similar:
                print(f"  - {word}")

# 1. TODO: What does the find_similiar_words method and print_simliar_words method do?
# 

# 
# Answer: response here

# ### Training Objective
# 
# For training a language model, the objective is for the model to assign high probability to the correct next word among all possible candidates in the vocabulary. In order to measure how far part the predicted distribution is from the ground-truth, we use cross entropy loss.
# 
# Cross entropy loss is defined as the negative log-probability of the correct word.
# 
# Intuitively:
# - If the model assigns high probability to the correct word, the loss is small.
# - If the model assigns low probability, the loss is large.
# 
# Formally, if the model outputs logits of shape $(B, V)$ where
# - $B = $ batch size,
# - $V = $ vocabulary size,
# - $y_i = $ the index of the correct word for example $i$,
# 
# then the loss is
# 
# $$
# \text{CrossEntropyLoss}(\text{logits}, \text{targets})
# = -\frac{1}{B} \sum_{i=1}^{B}
# \log \frac{\exp\big(\text{logits}_{i,\, y_i}\big)}
# {\sum_{j=1}^{V} \exp\big(\text{logits}_{i,j}\big)}
# $$
# where $j$ runs over all vocabulary items.

# In[ ]:


#2. TODO: Implement cross entropy loss using F.log_softmax and torch.gather

def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss using log and gather.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        targets: Tensor of shape (batch_size,), containing correct word indices

    Returns:
        loss: Scalar tensor, mean cross-entropy loss over the batch
    """
    # TODO: Implement log softmax over the vocabulary dimension
    log_probs = FILL_IN

    # TODO: Select the log-probabilities corresponding to the target words
    chosen_log_probs = FILL_IN

    # TODO: Take the mean negative log likelihood
    loss = FILL_IN

    return loss

# In[ ]:


# 3. TODO: fill in the missing parts of train_model
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def evaluate(model, data_loader, max_batches=None):
    """
    Average cross-entropy loss and perplexity on held-out data. (Provided, no work required.)
    """
    was_training = model.training
    model.eval()
    total_loss, total_examples = 0.0, 0
    with torch.no_grad():
        for b, (context_indices, target_idx) in enumerate(data_loader):
            if max_batches is not None and b >= max_batches:
                break
            context_indices = context_indices.to(model.device)
            target_idx = target_idx.to(model.device)
            logits = model(context_indices)
            total_loss += F.cross_entropy(logits, target_idx, reduction='sum').item()
            total_examples += target_idx.size(0)
    if was_training:
        model.train()
    avg_loss = total_loss / total_examples
    return avg_loss, math.exp(avg_loss)

def train_model(model, train_loader, learning_rate, tokenizer=None,
                max_steps=3000, experiment_name=None,
                val_loader=None, eval_every=1000, eval_batches=100):
    """
    Train model with simple max_steps termination.
    """
    #Creates unique experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"

    #Initializes TensorBoard writer
    log_dir = f"runs/{experiment_name}"
    if os.path.exists(log_dir):
        n = 2
        while os.path.exists(f"{log_dir}_run{n}"):
            n += 1
        log_dir = f"{log_dir}_run{n}"
        print(f"Log dir already exists; logging to {log_dir} instead")
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.001, total_iters=max_steps
    )

    acc_loss = 0.0
    acc_count = 0
    global_step = 0
    best_loss = float('inf')
    val_loss, val_ppl = None, None

    writer.add_hparams({
        'learning_rate': learning_rate,
        'max_steps': max_steps,
        'batch_size': train_loader.batch_size,
        'model_params': sum(p.numel() for p in model.parameters()),
    }, {})

    t0 = time.time()
    model.train()

    print(f"Training for {max_steps:,} steps...")

    train_iter = iter(train_loader)

    while global_step < max_steps:
        try:
        # TODO:
            context_indices, target_idx = FILL_IN
        except StopIteration:
            train_iter = iter(train_loader)
            context_indices, target_idx = FILL_IN

        context_indices = context_indices.to(model.device)
        target_idx = target_idx.to(model.device)

        # TODO: forward pass
        logits = None

        # TODO: compute loss
        loss = None

        # TODO: backward pass
        # (zero gradients, backpropagate, optimizer step, scheduler step)


        global_step += 1
        acc_loss += loss.item()
        acc_count += 1

        # Log to TensorBoard every step
        writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

        if global_step % 100 == 0:
            avg_loss = acc_loss / acc_count
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            progress = global_step / max_steps * 100

            print(f"[Step {global_step:,}/{max_steps:,} ({progress:.1f}%)] Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

            writer.add_scalar('Loss/Train_Avg100', avg_loss, global_step)

            if avg_loss < best_loss:
                best_loss = avg_loss
            writer.add_scalar('Best_Loss', best_loss, global_step)

            acc_loss = 0.0
            acc_count = 0

            if tokenizer is not None and global_step % 500 == 0:
                try:
                    query_words = ["linux", "python", "code", "system", "file"]
                    print_similar_words(model, tokenizer, query_words, k=3)
                except Exception as e:
                    print(f"Similarity check failed: {e}")

        # Evaluate on the validation set occasionally
        if val_loader is not None and global_step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, max_batches=eval_batches)
            writer.add_scalar('Loss/Val', val_loss, global_step)
            writer.add_scalar('Perplexity/Val', val_ppl, global_step)
            print(f"[Step {global_step:,}] Val loss: {val_loss:.4f} | Val perplexity: {val_ppl:.2f}")

        # Save checkpoint occasionally
        if global_step % 2000 == 0:
            checkpoint_path = f"model_checkpoint_step_{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    if val_loader is not None and global_step % eval_every != 0:
        val_loss, val_ppl = evaluate(model, val_loader, max_batches=eval_batches)
        writer.add_scalar('Loss/Val', val_loss, global_step)
        writer.add_scalar('Perplexity/Val', val_ppl, global_step)

    writer.close()

    final_time = time.time() - t0
    print(f"\nTraining completed!")
    print(f"Total steps: {global_step:,}")
    print(f"Total time: {final_time:.1f}s")
    print(f"Best loss achieved: {best_loss:.4f}")
    if val_loss is not None:
        print(f"Final val loss: {val_loss:.4f} | Final val perplexity: {val_ppl:.2f}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"To view: tensorboard --logdir={log_dir}")

    return {
        'final_step': global_step,
        'final_loss': best_loss,
        'total_time': final_time,
        'experiment_name': experiment_name,
        'val_loss': val_loss,
        'val_perplexity': val_ppl,
    }

# Training Setup

# The preprocessing first **tokenizes each document**, optionally truncates it to a maximum length, and appends an **end-of-sequence (EOS)** token to mark document boundaries.  
# All tokenized documents are **concatenated into a single 1-D array of token IDs**, forming one long text stream.
# 
# The `PackedLMDataset` then **slides a fixed-length context window** of size `T` across this stream.  
# For each valid position, it outputs:
# * **x** – the next `T` tokens (the context)
# * **y** – the single token immediately following that context (the prediction target)
# 
# This setup trains the model to predict the next token given a rolling window of prior tokens, even across document boundaries.
# 
# Preprocessing for 100k documents should be sufficient for training, expect it to take around 6 minutes on colab. No work required here.
# 

# In[ ]:


from tqdm import tqdm
class PackedLMDataset(Dataset):
    """
    token_ids: 1-D torch.LongTensor of packed tokens (… doc + EOS + doc + EOS …)
    context_size: length T of the context window.
    Returns (x:[T], y:int)
    """
    def __init__(self, token_ids: torch.Tensor, context_size: int):
        assert token_ids.dim() == 1
        self.ids = token_ids
        self.T = int(context_size)
        self.N = max(0, len(self.ids) - self.T)  # valid start positions

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        x = self.ids[i : i + self.T]
        y = self.ids[i + self.T]
        return x.long(), y.long()

def pack_rows(rows, tokenizer, max_doc_tokens, eos_id):
    """
    Tokenize each row, truncate to max_doc_tokens, append EOS, and concatenate.
    Returns a 1D numpy array of dtype int64.
    """
    pieces = []
    append = pieces.append

    for r in tqdm(rows, total=len(rows)):
        t = r.get("text") if isinstance(r, dict) else r["text"]
        if not t:
            continue

        ids = tokenizer.encode(t)
        if not ids:
            continue

        if max_doc_tokens is not None:
            ids = ids[:max_doc_tokens]

        append(np.asarray(ids, dtype=np.int64))
        append(np.asarray([eos_id], dtype=np.int64))

    if not pieces:
        return np.zeros((1,), dtype=np.int64)

    return np.concatenate(pieces, axis=0)

# In[ ]:


# Hyperparameters and dataset loading
CONTEXT = 10
MAX_DOC_TOKENS = 2500
VAL_FRACTION = 0.05
EOS_ID = len(tok.id_to_tok)
VOCAB_SIZE = len(tok.id_to_tok) + 1
NUM_DOCS = 100000

print("Loading dataset...")
ds_full = datasets.load_dataset('coms4705-hewitt/fineweb-linuxlike', 'default', split='train')
ds_subset = ds_full.shuffle(seed=123).select(range(NUM_DOCS))
splits = ds_subset.train_test_split(test_size=VAL_FRACTION, seed=123)
train_rows, val_rows = splits['train'], splits['test']

print("Packing training data...")
train_ids_np = pack_rows(train_rows, tok, MAX_DOC_TOKENS, EOS_ID)
print("Packing validation data...")
val_ids_np = pack_rows(val_rows, tok, MAX_DOC_TOKENS, EOS_ID)

train_ids = torch.from_numpy(train_ids_np)
val_ids = torch.from_numpy(val_ids_np)

train_ds = PackedLMDataset(train_ids, context_size=CONTEXT)
val_ds = PackedLMDataset(val_ids, context_size=CONTEXT)

train_loader = DataLoader(
    train_ds, batch_size=512, shuffle=True,
    num_workers=0, drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=512, shuffle=False,
    num_workers=0, drop_last=False
)

print(f"Packed tokens: train={len(train_ids):,} val={len(val_ids):,}")
print(f"Training examples: {len(train_ds):,}")
print(f"Validation examples: {len(val_ds):,}")
print(f"Vocab size for the model (include EOS): {VOCAB_SIZE}")

# In[ ]:


EMBEDDING_DIM = 100
learning_rate = 0.001

# # **Time to train your first model!**
# 
# notes: Please do not change the model names [single_vector_model, positional_model, mlpdeep, residualmlp, gatedresidualmlp]

# In[ ]:


#TRAINING THE MODEL (should approximately take 2 minutes on the T4 GPU for 3000 steps)
single_vector_model = SingleVectorWordModel(VOCAB_SIZE, EMBEDDING_DIM)
single_vector_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
single_vector_model.to(single_vector_model.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in single_vector_model.parameters()])))

history = train_model(single_vector_model, train_loader, learning_rate, tokenizer=tok, max_steps=3000, experiment_name = "SingleVectorWordModel1", val_loader=val_loader)

# 4. TODO: You can use Tensorboard like this to visualize your training metrics

# In[ ]:


%load_ext tensorboard
%tensorboard --logdir runs

# 5. TODO: Try training the model multiple times. What do you notice about the loss of the first step across multiple training runs. Does this surprise you? Why or why not?
# 
# Answer: response here

# # **Improving the model**
# Let's improve the model by allowing it to capture the positional meaning in a text sequence.
# $$
# h_{< i} =
# \sigma \left(
#     \frac{1}{\,i-1\,}
#     \sum_{j=1}^{\,i-1}
#         \sigma\big(E x_j + p_j\big)
# \right)
# $$
# 
# Task: Complete the TODO: sections in the NonLinearPositionalContextModel class

# In[ ]:


class NonLinearPositionalContextModel(SingleVectorWordModel):
    def __init__(self, vocab_size, embedding_dim, max_context):
        super().__init__(vocab_size, embedding_dim)
        #-------Begin TODO-------
        self.max_context = FILL_IN
        self.pos = nn.Embedding(FILL_IN, FILL_IN)
        nn.init.uniform_(FILL_IN, FILL_IN, FILL_IN)
        self.sigmoid = FILL_IN
        #-------End TODO-----------

    def encode_context(self, context_indices):
        #--------Begin TODO--------
        return FILL_IN
        #--------End TODO----------


# In[ ]:


#TRAINING THE MODEL
positional_model = NonLinearPositionalContextModel(VOCAB_SIZE, EMBEDDING_DIM, max_context=10)
positional_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
positional_model.to(positional_model.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in positional_model.parameters()])))

history = train_model(positional_model, train_loader, learning_rate, tokenizer=tok, max_steps=3000, experiment_name = "NonLinearPositionalContextModel1", val_loader=val_loader)

# ### From averaging to attention
# 
# The positional model gives every context token the same weight, $1/L$. Attention replaces
# that equal-weight average with a weighted one, where each token decides how much of every
# other token to pull in — tokens that are similar pull more from each other. See Lecture 3,
# "Information flow and simple attention," for the derivation.
# 
# Let $Z \in \mathbb{R}^{B \times L \times d}$ be the token representations the positional
# model already builds, $z_j = \sigma(Ex_j + p_j)$. The whole operation is three lines:
# 
# $$e = \frac{ZZ^\top}{\sqrt{d}} \qquad \alpha = \text{softmax}(e) \qquad H = \alpha Z$$
# 
# with $e, \alpha \in \mathbb{R}^{B \times L \times L}$ and $H \in \mathbb{R}^{B \times L \times d}$.
# Remember that $ZZ^\top$ computes $e_{bij} = z_{bi}^\top z_{bj}$, and the softmax is taken over the **last** axis —
# of the two length-$L$ axes, the first indexes the token doing the attending and the second
# the tokens being attended to. The $\sqrt{d}$ keeps the scores from growing with $d$; more
# on that in Lecture 8.
# 
# Task: Complete the TODO: sections in the SimplifiedAttentionContextModel class

# In[ ]:


class SimplifiedAttentionContextModel(NonLinearPositionalContextModel):
    """
    Same embeddings, positions, nonlinearity, and scoring as the positional model.
    The only change: the equal-weight average becomes a similarity-weighted one.
    """
    def encode_context(self, context_indices):
        B, L = context_indices.size()
        we = self.embeddings1(context_indices)                               # (B, L, d)
        pe = self.pos(torch.arange(L, device=context_indices.device))[None]  # (1, L, d)
        Z = self.sigmoid(we + pe)                                            # (B, L, d)
        # Note that each score is a sum of d terms. What happens to the softmax if d grows and we don't scale?
        scale = Z.size(-1) ** 0.5   # keeps scores from growing with d.

        #--------Begin TODO--------
        # 1) all pairwise similarities e[b, i, j] = Z[b, i] . Z[b, j],
        #    divided by scale                                             -> (B, L, L)
        e = FILL_IN
        # 2) normalize over the tokens being attended to, the last axis   -> (B, L, L)
        alpha = FILL_IN
        # 3) weighted sum H[b, i] = sum_j alpha[b, i, j] * Z[b, j]         -> (B, L, d)
        H = FILL_IN
        #--------End TODO----------

        self.last_alpha = alpha.detach()   # saved so you can inspect the weights below
        return self.sigmoid(H[:, -1])      # last position's representation, (B, d)

# In[ ]:


#TRAINING THE MODEL
attention_model = SimplifiedAttentionContextModel(VOCAB_SIZE, EMBEDDING_DIM, max_context=10)
attention_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attention_model.to(attention_model.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in attention_model.parameters()])))

history = train_model(attention_model, train_loader, learning_rate, tokenizer=tok, max_steps=3000, experiment_name = "SimplifiedAttentionContextModel1", val_loader=val_loader)

# In[ ]:


# The attention weights are saved in model.last_alpha during the forward pass.
# last_alpha[b, i, j] is how much position i attended to position j, for example b.
# We print row -1, since that is the row actually used to make the prediction.
ctx, target = next(iter(train_loader))
ctx = ctx.to(attention_model.device)
with torch.no_grad():
    attention_model(ctx)

for b in range(3):
    tokens = [tok.decode([i]) for i in ctx[b].tolist()]
    weights = attention_model.last_alpha[b, -1].tolist()
    print('context: ' + repr(''.join(tokens)) + '  ->  next: ' + repr(tok.decode([target[b].item()])))
    for t, w in zip(tokens, weights):
        print(f'  {w:6.3f}  {t!r}')
    print()

# ### From Weighted Averages to a Context-Level MLP
# 
# Attention got the tokens mixing, but notice what it *didn't* do: the weights $\alpha$ are
# just dot products of the token representations with each other, so there are no
# parameters in that mixing at all.
# And however the weights come out, the result is still a weighted **sum** of per-token
# features, which the final predictor then has to work with.
# 
# Let's try removing that pooling bottleneck a different way.  
# What if we embed each token, concatenate the entire fixed-length context
# (`L x d` → `L*d`), and feed this single vector through a multi-layer
# perceptron before scoring.  
# Because the MLP operates on the *whole* context vector, it can freely mix
# dimensions from different positions in every layer.
# 
# #### The motivation for this change
# * **Broader function class.**  
#   For a fixed context length, a sufficiently wide MLP can approximate any
# mapping from the concatenated embeddings to the output, while the positional model was limited to additive “bag-of-tokens” functions.
# 
# * **Position awareness without extra machinery.**  
#   The concatenation preserves a consistent slot order
# (position 1 occupies the first block of `d` features, position 2 the next, etc.).
# The weight matrices naturally learn different transformations for each slot,
# so no separate positional embeddings are required.
# 
# This architecture is still simpler than a Transformer-there is no self-attention
# or dynamic token-token interaction but it is a meaningful next step in
# expressivity beyond token-wise pooling.
# 
# Task: Complete the TODO: sections in the FeedForwardContextModel class
# 

# In[ ]:


import torch
import torch.nn as nn

class FeedForwardContextModel(SingleVectorWordModel):
    """
    Context-level MLP (no positional embeddings):
      tokens -> embed -> flatten (B, L*d) -> MLP -> score
    This model treats the concatenated embeddings of all L tokens
    as a single input vector, allowing cross-token interactions.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 ff_hidden_dim: int,
                 max_context: int):
        super().__init__(vocab_size, embedding_dim)
        self.max_context = int(max_context)

        # TODO: define MLP: (L*d) -> hidden -> d
        self.ff = None

        # TODO: initialize MLP weights (e.g., xavier) and biases to zero
        # for m in self.ff:
        #     if isinstance(m, nn.Linear):
        #         ...

    def forward(self, context_indices: torch.LongTensor) -> torch.Tensor:
        """
        context_indices: (B, L) with L == max_context
        """
        B, L = context_indices.shape
        assert L == self.max_context, f"Expected context length {self.max_context}, got {L}"

        # TODO: 1) Embed tokens -> (B, L, d)
        x = None

        # TODO: 2) Flatten entire context -> (B, L*d)
        x_flat = None

        # TODO: 3) Context-level MLP -> (B, d)
        h = None

        # TODO: 4) Score with tied vocabulary embeddings -> (B, V)
        logits = None

        return logits


# In[ ]:


HIDDEN_DIM = 1028
mlp = FeedForwardContextModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, 10)
mlp.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp.to(mlp.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in mlp.parameters()])))

history = train_model(mlp, train_loader, learning_rate, tokenizer=tok, max_steps=10000, experiment_name = "FeedForwardContextModel1", val_loader=val_loader)


# ### Going Deeper
# 
# The current context level mlp applies a **single feed-forward block**
# to the flattened `L*d` context vector before prediction.  
# A natural next question is: *how does the model’s **expressive power** change if
# we stack multiple such blocks?*
# 
# In function-approximation terms, a one-layer MLP represents functions of the form
# 
# $$
# f(x) = \sigma(Wx + b)
# $$
# 
# where \(\sigma\) is a fixed non-linearity.  
# Adding more layers composes these transformations:
# 
# $$
# f(x) = W_k \,\sigma\big(W_{k-1}\,\sigma(\dots \sigma(W_1 x)\dots )\big)
# $$
# 
# Classical results in approximation theory show that **compositions of simple
# nonlinearities can approximate a far richer class of functions** than a single
# layer of the same width.  
# Unlike the positional model, the input here is already the full context vector,
# so deeper stacks directly expand the space of context-level mappings the model
# can learn.
# 
# 
# #### Task
# 1. Add a `num_layers` parameter so the feed-forward block can be repeated
#    multiple times.  
#    Begin with `num_layers = 1` (the current model) and experiment with `2`, `4`,
#    and `8`.
# 2. Compute how the parameter count scales with depth.
# 3. Train each version under identical hyper-parameters and compare:
#    * training loss and validation perplexity
#    * gradient norms in each layer (use `print_grad_stats`) to monitor stability.
# 
# 
# 
# #### Things to Observe
# * **Expressivity vs. optimization** – deeper networks can capture more complex
# context functions but may be harder to train.
# * **Gradient flow** – check for vanishing or exploding gradients as depth grows.
# * **Parameter efficiency** – assess whether additional layers provide measurable
# gains relative to their extra parameters and compute.
# 

# In[ ]:


def print_grad_stats(model):
    """Print L2 norms of gradients for each layer."""
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name:<30s} grad_norm={p.grad.norm().item():.6f}")

# In[ ]:


class _ContextMLPBlock(nn.Module):
    """
    Minimal width-preserving block operating on a d-dim vector:
      d -> hidden -> d (with an activation in between).
    No residuals, no LayerNorm—keeps the step small.
    """
    def __init__(self, d: int, hidden: int, activation=nn.ReLU):
        super().__init__()
        # TODO: define first linear layer (d -> hidden)
        self.fc1 = None
        # TODO: activation function
        self.act = None
        # TODO: define second linear layer (hidden -> d)
        self.fc2 = None
        # TODO: initialize weights (e.g., xavier) and biases to zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward pass through fc1 -> activation -> fc2
        return None


class FeedForwardContextModelDeep(SingleVectorWordModel):
    """
    Context-level MLP (no positional embeddings):
      tokens -> embed -> flatten (B, L*d) -> stem -> [block] * num_layers -> score

    - Unlike the original FeedForwardContextModel, the stem first maps (L*d) -> d,
      and num_layers blocks of d -> hidden -> d are applied on top.
    - Increasing num_layers deepens only the d→hidden→d mapping.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,   # d
                 ff_hidden_dim: int,   # hidden width inside each block
                 max_context: int,     # L
                 num_layers: int = 1,
                 activation=nn.ReLU):
        super().__init__(vocab_size, embedding_dim)
        self.max_context = int(max_context)
        self.num_layers = int(num_layers)
        d = embedding_dim
        L = self.max_context

        # TODO: Stem layer: (L*d) -> d
        self.stem_fc = None
        self.stem_act = None
        # TODO: initialize stem weights and biases

        # TODO: create ModuleList of num_layers ContextMLPBlocks
        #       (pass in all params, e.g. _ContextMLPBlock(d, ff_hidden_dim, activation))
        self.blocks = None

    def forward(self, context_indices: torch.LongTensor) -> torch.Tensor:
        """
        context_indices: (B, L) with L == max_context
        """
        B, L = context_indices.shape
        assert L == self.max_context, f"Expected context length {self.max_context}, got {L}"

        # TODO: 1) Embed tokens -> (B, L, d)
        x = None

        # TODO: 2) Flatten entire context -> (B, L*d)
        x_flat = None

        # TODO: 3) Stem to d -> (B, d). Remember to apply the stem activation.
        h = None

        # TODO: 4) Depth: apply each block (d -> hidden -> d)
        # for blk in self.blocks:
        #     h = blk(h)

        # TODO: 5) Tied scoring -> (B, V)
        logits = None

        return logits


# In[ ]:


#approximately 10 mins to train for 4 layers of hidden_dim = 1028, 15k steps.
HIDDEN_DIM = 1028
mlpdeep = FeedForwardContextModelDeep(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, max_context=10, num_layers = 4)
mlpdeep.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlpdeep.to(mlpdeep.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in mlpdeep.parameters()])))

history = train_model(mlpdeep, train_loader, learning_rate, tokenizer=tok, max_steps=15000, experiment_name = "FeedForwardContextModelDeep1", val_loader=val_loader)


# In[ ]:


print_grad_stats(mlpdeep)

# ### Residual Connections
# 
# When we stack many fully-connected layers, training can slow down or become
# unstable because gradients shrink or grow as they pass through each layer.
# A **residual connection** helps by adding the input of a block directly to its
# output:
# 
# $$
# \text{Block}(x) = x + f(x)
# $$
# 
# where \(f(x)\) is the small network inside the block.
# 
# **Why this helps**
# * The gradient can flow along the simple identity path \(x \rightarrow x\),
#   so learning is less sensitive to depth or initialization.
# * Each block only needs to learn a *correction* \(f(x)\) on top of the input,
#   instead of the entire mapping from scratch.
# 
# #### Tasks
# 
# 1. **Implement a `ResidualBlock`**
#    * Input and output both have size `d`.
#    * Inside the block: `Linear(d→h)` → activation (e.g., ReLU) → `Linear(h→d)`
#      to compute \(f(x)\).
#    * Return `x + f(x)`.
# 
# 2. **Build a `ResidualFeedForwardModel`**
#    * Start from your context-level MLP pipeline
#      (`embed → flatten → stem` to dimension `d`).
#    * Stack several `ResidualBlock`s in a `ModuleList`.
#    * After the stack, use the same `score()` method to produce logits.
# 
# 3. **Experiment**
#    * Train deep models with and without residuals using the same settings.
#    * Plot training and validation loss to compare convergence speed.
#    * Use `print_grad_stats` to inspect gradient norms across layers.
# 

# In[ ]:


class ResidualBlock(nn.Module):
    """
    Width-preserving residual block on a d-dim vector:
      x -> Linear(d->hidden) -> Act -> Linear(hidden->d) -> + x
    """
    def __init__(self, d: int, hidden: int, activation=nn.ReLU, use_layernorm: bool = False):
        super().__init__()
        self.use_layernorm = use_layernorm
        # TODO: optional LayerNorm or Identity
        self.ln = None

        # TODO: first linear layer (d -> hidden)
        self.fc1 = None
        # TODO: activation function
        self.act = None
        # TODO: second linear layer (hidden -> d)
        self.fc2 = None
        # TODO: initialize weights (e.g., xavier) and zero biases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: optional pre-norm, forward through fc1 -> activation -> fc2, add residual
        return None


class ResidualFeedForwardContextModel(SingleVectorWordModel):
    """
    Context-level residual MLP:
      tokens -> embed -> flatten (B, L*d) -> stem (L*d -> d)
              -> [ResidualBlock(d, hidden)] * num_layers
              -> score (tied embeddings)
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,   # d
                 max_context: int,     # L
                 hidden_dim: int,      # hidden width inside each residual block
                 num_layers: int = 2,  # number of residual blocks
                 activation=nn.ReLU,
                 use_layernorm: bool = False):
        super().__init__(vocab_size, embedding_dim)
        self.max_context = int(max_context)
        d = embedding_dim
        L = self.max_context

        # TODO: Stem (L*d -> d) with activation
        self.stem = None
        # TODO: initialize stem weights/biases

        # TODO: stack of residual blocks
        #       (pass in all params, e.g. ResidualBlock(d, hidden_dim, activation, use_layernorm))
        self.blocks = None

    def forward(self, context_indices: torch.LongTensor) -> torch.Tensor:
        """
        context_indices: (B, L) with L == max_context
        """
        B, L = context_indices.shape
        assert L == self.max_context, f"Expected context length {self.max_context}, got {L}"

        # TODO: 1) Embed tokens -> (B, L, d)
        x = None

        # TODO: 2) Flatten entire context -> (B, L*d)
        x_flat = None

        # TODO: 3) Stem to d -> (B, d). Remember to apply the stem activation.
        h = None

        # TODO: 4) Apply residual blocks
        # for blk in self.blocks:
        #     h = blk(h)

        # TODO: 5) Tied scoring -> (B, V)
        logits = None

        return logits


# In[ ]:


residualmlp = ResidualFeedForwardContextModel(VOCAB_SIZE, EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, max_context=10, num_layers=4)
residualmlp.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
residualmlp.to(residualmlp.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in residualmlp.parameters()])))

history = train_model(residualmlp, train_loader, learning_rate, tokenizer=tok, max_steps=15000, experiment_name = "ResidualFeedForwardContextModel1", val_loader=val_loader)


# ### Gating
# 
# Gating is a learned, elementwise mask that modulates a signal.  
# Given a representation \(x\), a gate \(g(x)\) produces per-dimension weights in \([0,1]\), and the modulated output is
# $$
# y \;=\; \sigma(g(x)) \odot h(x)
# $$
# where \(h(x)\) is any transformation of \(x\), \(\sigma\) is a sigmoid, and \(\odot\) is elementwise multiply.
# 
# **Why use it**
# - **Selective flow:** the model can amplify useful features and suppress noisy ones.
# - **Dynamic sparsity:** encourages the network to use only the channels it needs for a given input.

# In[ ]:


class GatedResidualBlock(nn.Module):
    """
    Width-preserving gated residual block on a d-dim vector:
      y = x + sigmoid(g(x)) * f(x)
      where f(x): d -> h -> d, g(x): d -> d
    """
    def __init__(self, d: int, h: int, activation=nn.ReLU, use_layernorm: bool=False):
        super().__init__()
        # TODO: optional LayerNorm or Identity
        self.ln = None

        # TODO: define f1 (d -> h), activation, f2 (h -> d)
        self.f1 = None
        self.act = None
        self.f2 = None

        # TODO: define gating layer g (d -> d)
        self.g = None

        # TODO: initialize all weights/biases (e.g., xavier) and set gate bias to start closed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply layernorm (if used), compute f(x), compute gate(x),
        #       and return x + sigmoid(gate) * f(x)
        return None


class GatedResidualFeedForwardContextModel(SingleVectorWordModel):
    """
    Context-level gated residual MLP (no positional embeddings):
      tokens -> embed -> flatten (B, L*d) -> stem (L*d -> d)
             -> [GatedResidualBlock(d,h)] * num_layers
             -> score()  (tied embeddings from SingleVectorWordModel)
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,   # d
                 max_context: int,     # L
                 hidden_dim: int,      # h inside each block
                 num_layers: int = 2,
                 activation=nn.ReLU,
                 use_layernorm: bool=False):
        super().__init__(vocab_size, embedding_dim)
        self.max_context = int(max_context)
        d, L = embedding_dim, self.max_context

        # TODO: Stem layer (L*d -> d) with activation
        self.stem = None
        # TODO: initialize stem weights/biases

        # TODO: stack of gated residual blocks
        #       (pass in all params, e.g. GatedResidualBlock(d, hidden_dim, activation, use_layernorm))
        self.blocks = None

    def forward(self, context_indices: torch.LongTensor) -> torch.Tensor:
        """
        context_indices: (B, L) with L == max_context
        """
        B, L = context_indices.shape
        assert L == self.max_context, f"Expected context length {self.max_context}, got {L}"

        # TODO: 1) Embed tokens -> (B, L, d)
        x = None

        # TODO: 2) Flatten -> (B, L*d)
        x_flat = None

        # TODO: 3) Stem -> (B, d). Remember to apply the stem activation.
        h = None

        # TODO: 4) Apply each gated residual block
        # for blk in self.blocks:
        #     h = blk(h)

        # TODO: 5) Tied scoring -> (B, V)
        logits = None

        return logits


# In[ ]:


gatedresidualmlp = GatedResidualFeedForwardContextModel(VOCAB_SIZE, EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, num_layers=4, max_context=10)
gatedresidualmlp.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gatedresidualmlp.to(gatedresidualmlp.device)

print('Model has {} parameters'.format(sum([np.prod(p.size()) for p in gatedresidualmlp.parameters()])))

history = train_model(gatedresidualmlp, train_loader, learning_rate, tokenizer=tok, max_steps=15000, experiment_name = "GatedResidualFeedForwardContextModel1", val_loader=val_loader)

# # **Take a look at how the models compare!**

# In[ ]:


%load_ext tensorboard
%tensorboard --logdir runs

# Discuss the results that you gathered throughout this assignment in two short paragraphs. You might consider topics such as:
# 
# 1. How the different model architectures performed relative to each other.
# 
# 2. Trends you observed in training (loss curves, convergence speed, overfitting, etc.).
# 
# 3. The impact of hyper-parameters.
# 
# 4. Any unexpected behaviors or challenges you encountered.
# 
# Support your discussion with specific observations or metrics where possible.

# Response here:
# 
