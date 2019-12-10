import sys
from FormatSSVAE.vae import FormatVAE

NUM_GENERATION = sys.argv[2]
vae = FormatVAE(encoder_hidden_size=256, decoder_hidden_size=64, mlp_hidden_size=32)
vae.load_checkpoint(filename=sys.argv[1].split('/')[-1])

print("================")
print(f"Generated Names")
print("================")
for _ in range(int(NUM_GENERATION)): print(f"- {vae.model(None)[0]}")
