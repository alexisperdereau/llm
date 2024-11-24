vocab_size = 100  # Par exemple, 100 mots
seq_len = 10      # Longueur des séquences
batch_size = 2    # Pour des raisons de mémoire

model = Transformer(vocab_size, d_model=32, num_heads=2, num_layers=2, dim_feedforward=64, max_len=seq_len)
input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
output = model(input_seq)
print(output.shape)  # Devrait être [batch_size, seq_len, vocab_size]
