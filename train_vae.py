import pandas as pd
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import poutine
import sys
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from FormatSSVAE.vae import FormatVAE
from FormatSSVAE.util.name_ds import NameDataset
from FormatSSVAE.util.plot import plot_losses


pyro.enable_validation(True)
NUM_EPOCHS = 1000
ADAM_CONFIG = {'lr': 0.0005}
BATCH_SIZE = 2048
MAX_INPUT_STRING_LEN = 18

def weights_for_balanced_class(df, target_column):
    """
    Assign higher weights to rows whose class is not prevalent
    to sample each class equally with DataLoader
    """
    target = df
    num_classes = target.nunique()
    counts = [0] * num_classes
    for row_class in target: counts[row_class] += 1
    class_weights = [0] * num_classes
    for i,count in enumerate(counts): class_weights[i] = len(target)/count
    weights = [0] * len(target)
    for i,row_class in enumerate(target): weights[i] = class_weights[row_class]
    return weights

def kl_annealing(model, guide, *args, **kwargs):
    # get the annealing factor and latents to anneal from the keyword
    # arguments passed to the model and guide
    annealing_factor = kwargs.pop('annealing_factor', 1.0)
    latents_to_anneal = kwargs.pop('latents_to_anneal', [])
    # run the guide and replay the model against the guide
    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(
        poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    elbo = 0.0
    # loop through all the sample sites in the model and guide trace and
    # construct the loss; note that we scale all the log probabilities of
    # samples sites in `latents_to_anneal` by the factor `annealing_factor`
    for site in model_trace.values():
        if site["type"] == "sample":
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            elbo = elbo + factor * site["fn"].log_prob(site["value"]).sum()
    for site in guide_trace.values():
        if site["type"] == "sample":
            factor = annealing_factor if site["name"] in latents_to_anneal else 1.0
            elbo = elbo - factor * site["fn"].log_prob(site["value"]).sum()
    return -elbo

def train_one_epoch(loss, dataloader, epoch_num, model, guide):
    total_loss = 0.
    i = 1
    for batch in dataloader:
        batch_loss = loss.step(model(batch), guide(batch), annealing_factor=0.2, latents_to_anneal=["my_latent"])/len(batch)
        total_loss += batch_loss
        if i%10 == 0: print(f"Epoch {epoch_num} {i}/{len(dataloader)} Loss: {batch_loss}")
        i += 1
        
    avg_loss = total_loss/len(dataloader)
    return avg_loss

dataset = NameDataset("data/cleaned.csv", "name", max_string_len=MAX_INPUT_STRING_LEN, format_col_name='format')
sample_weights = weights_for_balanced_class(dataset.format_col, 'format')
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)

vae = FormatVAE(encoder_hidden_size=256, decoder_hidden_size=64, mlp_hidden_size=32)
if len(sys.argv) > 1: vae.load_checkpoint(filename=sys.argv[1].split('/')[-1])
svi_loss = SVI(vae.model, vae.guide, Adam(ADAM_CONFIG), loss=kl_annealing)

epoch_losses = []
for e in range(NUM_EPOCHS):
    print("===========================")
    print(f"Epoch {e} Generated Names")
    print("===========================")
    for _ in range(5): print(f"- {vae.model(None)[0]}")
    avg_loss = train_one_epoch(svi_loss, dataloader, e, vae.model, vae.guide)
    vae.save_checkpoint(filename="test.pth.tar")
    epoch_losses.append(avg_loss)
    plot_losses(epoch_losses, filename="test.png")
