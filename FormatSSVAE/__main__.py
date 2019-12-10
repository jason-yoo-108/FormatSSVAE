from FormatSSVAE.vae import FormatVAE

if __name__ == "__main__":
    vae = FormatVAE(decoder_hidden_size=8)
    print(vae.model(["jason","ellen"]))
    print(vae.model(None))