import pandas as pd
import torch
import numpy as np

embeddings_df = pd.read_csv('embeddings.csv')

embeddings_np = embeddings_df.to_numpy()
embeddings = torch.tensor(embeddings_np, dtype=torch.float)
names_df = pd.read_csv('names.csv')

names_np = names_df['Name'].to_numpy()
names = names_np.tolist()

print(embeddings.shape)
print(names)
