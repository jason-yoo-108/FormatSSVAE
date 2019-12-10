import pandas as pd
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.utils.data import DataLoader

from FormatSSVAE.vae import FormatVAE
from FormatSSVAE.util.name_ds import NameDataset
from FormatSSVAE.util.plot import plot_losses


pyro.enable_validation(True)
NUM_EPOCHS = 100
ADAM_CONFIG = {'lr': 0.0005}
MAX_INPUT_STRING_LEN = 18

dataset = NameDataset("data/marathon_results_2015.csv", "Name", max_string_len=MAX_INPUT_STRING_LEN)
dataset.add_csv("data/marathon_results_2016.csv", "Name")
dataset.add_csv("data/marathon_results_2017.csv", "Name")

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
vae = FormatVAE(encoder_hidden_size=256, decoder_hidden_size=64, mlp_hidden_size=32)
svi_loss = SVI(vae.model, vae.guide, Adam(ADAM_CONFIG), loss=Trace_ELBO())

def train_one_epoch(loss, dataloader, epoch_num):
    total_loss = 0.
    i = 1
    for batch in dataloader:
        batch_loss = loss.step(batch)/len(batch)
        total_loss += batch_loss
        if i%10 == 0: print(f"Epoch {epoch_num} {i}/{len(dataloader)} Loss: {batch_loss}")
        i += 1
        
    avg_loss = total_loss/len(dataloader)
    return avg_loss

epoch_losses = []
for e in range(NUM_EPOCHS):
    print("===========================")
    print(f"Epoch {e} Generated Names")
    print("===========================")
    for _ in range(5): print(f"- {vae.model(None)[0]}")
    avg_loss = train_one_epoch(svi_loss, dataloader, e)
    vae.save_checkpoint(filename="test.pth.tar")
    epoch_losses.append(avg_loss)
    plot_losses(epoch_losses, filename="test.png")
