'''
   embedding_analysis: analysis embeddings produced from PLM.
   ref: https://github.com/swansonk14/language_models_biology/blob/main/Protein_Language_Model_Demo.ipynb
   created on Sep 26, 2024, mht
'''
import os
import time
import numpy as np
import seaborn as sns
import argparse
import pickle
import random
from pathlib import Path
from umap import UMAP
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process TM-Vec arguments', add_help=True)

parser.add_argument("--input-embedding",
                    type=Path,
                    required=False,
                    default="./example/result/embeddings.npy",
                    help=("Input embedding in pkl format to encode LSH.")
                    )

# load arguments
args = parser.parse_args()

embeddings = np.load(args.input_embedding)
print(embeddings.shape) # 10x512

# Apply UMAP to reduce the embeddings to 2 dimensions
embeddings_2d = UMAP(random_state=0).fit_transform(embeddings)
#print(embeddings_2d)
#plt.scatter(embeddings_2d[:,0],
 #           embeddings_2d[:,1])

sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1])
plt.xlabel("UMAP 0")
plt.ylabel("UMAP 1")
plt.xticks([])
plt.yticks([])

plt.title("NMPFamsDB Sequence Bacteria F000013")
plt.gca().set_aspect('equal', 'datalim') # setting scale y-unit/x-unit

plt.show()

#print(embeddings_2d.shape)
