# %%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from functools import partial
import json
from typing import List, Tuple, Union, Optional, Callable, Dict
import torch as t
from torch import Tensor
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import einops
from tqdm import tqdm
from jaxtyping import Float, Int, Bool
from pathlib import Path
import pandas as pd
import circuitsvis as cv
from IPython.display import display
from transformer_lens import utils, ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm
import pytorch_lightning as pl
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
import wandb
from pytorch_lightning.loggers import WandbLogger
import random

section_dir = os.getcwd()
from plotly_utils import hist, bar, imshow
from palindromes_datasets import SimpleTokenizer, PalindromesDataset

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"
# %%
VOCAB = "1234567890"

cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=3,
    n_layers=3,
    attention_dir="bidirectional", # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
    d_vocab_out=2, # 2 because we're doing binary classification
    use_attn_result=True, 
    device=device,
    use_hook_tokens=True
)

model_new = HookedTransformer(cfg).to(device)

# %%
N_SAMPLES = 5000
with open(section_dir / "outputs.json") as f:
    data_tuples: List[Tuple[str, bool]] = json.load(f)
    print(f"loaded {len(data_tuples)} examples")
assert isinstance(data_tuples, list)
random.shuffle(data_tuples)
data_train = PalindromesDataset(data_tuples[1000:]).to(device)
data_test = PalindromesDataset(data_tuples[:1000]).to(device)
# %%
# Plot sequence distribution of the data
hist(
    [len(x) for x, _ in data_tuples], 
    nbins=data_test.seq_length,
    title="Sequence lengths of brackets in dataset",
    labels={"x": "Seq len"}
)
# %%
tokenizer = SimpleTokenizer("1234567890")
def add_perma_hooks_to_mask_pad_tokens(model: HookedTransformer, pad_token: int) -> HookedTransformer:

    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(tokens: Float[Tensor, "batch seq"], hook: HookPoint) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_token, "b sK -> b 1 1 sK")

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model


model_new.reset_hooks(including_permanent=True)
model_new = add_perma_hooks_to_mask_pad_tokens(model_new, tokenizer.PAD_TOKEN)
# %%
model_new.to("cpu")
def run_model_on_data(model: HookedTransformer, data: PalindromesDataset, batch_size: int = 200) -> Float[Tensor, "batch 2"]:
    '''Return probability that each example is balanced'''
    all_logits = []
    for i in tqdm(range(0, len(data.strs), batch_size)):
        toks = data.toks[i : i + batch_size].to(device)
        logits = model(toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits


test_set = data_test[:1000]
n_correct = (run_model_on_data(model_new, test_set).argmax(-1).bool() == test_set.isbal).sum()
print(f"\nModel got {n_correct} out of {len(data_test)} training examples correct!")
# %%
cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=3,
    n_layers=3,
    attention_dir="bidirectional",
    act_fn="relu",
    d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
    d_vocab_out=2, # 2 because we're doing binary classification
    use_attn_result=True, 
    device=device,
    use_hook_tokens=True
)

model_new = HookedTransformer(cfg).to(device)

@dataclass
class TransformerTrainingArgs():
    batch_size = 8
    max_epochs = 6
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day1.4-transformer-palindrome"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


args = TransformerTrainingArgs()
data_test_loader = DataLoader(data_test, batch_size=64, shuffle=True)
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True)

class LitTransformer(pl.LightningModule):
    def __init__(self, args: TransformerTrainingArgs, model: HookedTransformer, data_train_loader: DataLoader, data_test_loader: DataLoader):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self.args = args
        self.data_train = data_train_loader
        self.data_test = data_test_loader
    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch 2"]:
        logits = self.model(tokens)
        return logits[:,0]
    
    def _shared_train_val_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        tokens, labels = batch[2].to(device), batch[1].to(device)
        logits = self(tokens)
        return logits, labels
    
    def training_step(self, batch: Tuple[Tensor,Tensor,Tensor], batch_idx: int) -> Float[Tensor, ""]:
        logits, labels = self._shared_train_val_step(batch)
        log_probs = -logits.log_softmax(dim=-1)
        loss = log_probs[t.arange(log_probs.shape[0]),labels.int()].mean()
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch: Tuple[Tensor,Tensor,Tensor], batch_idx: int) -> None:
        logits, labels = self._shared_train_val_step(batch)
        classifications = logits.argmax(dim=-1)
        accuracy = t.sum(classifications == labels) / len(classifications)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        optimizer = t.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def train_dataloader(self):
        return self.data_train
    def val_dataloader(self):
        return self.data_test
# %%
litmodel = LitTransformer(args, model_new, data_train_loader, data_test_loader)
logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)

trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    logger=logger,
    log_every_n_steps=args.log_every_n_steps
)
trainer.fit(model=litmodel, train_dataloaders=litmodel.data_train)
wandb.finish()

# %%
# Define and tokenize examples
examples = ["11", "55", "187", "5", "123454321", "23595496", "9965699"]
labels = [True, True, False, True, True, False, True]
toks = tokenizer.tokenize(examples)
model_new.to(device)
# Get output logits for the 0th sequence position (i.e. the [start] token)
logits = model_new(toks)[:, 0]

# Get the probabilities via softmax, then get the balanced probability (which is the second element)
prob_balanced = logits.softmax(-1)[:, 1]

# Display output
print("Model confidence:\n" + "\n".join([f"{ex:18} : {prob:<8.4%} : label={int(label)}" for ex, prob, label in zip(examples, prob_balanced, labels)]))
# %%
t.save(model_new.state_dict(), section_dir/"palindromes_model_state_dict.pt")
# %%
