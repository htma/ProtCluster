#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import encode
from transformers import T5EncoderModel, T5Tokenizer
#import gc
from pysam.libcfaidx import FastxFile
from pathlib import Path
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process TM-Vec arguments', add_help=True)
parser.add_argument("--input-fasta",
                    type=Path,
                    required=True,
                    help=("Input proteins in fasta format to "
                          "construct the database.")
)
parser.add_argument("--tm-vec-model",
                    type=Path,
                    required=True,
                    help="Model path for TM-Vec embedding model"
)
parser.add_argument("--tm-vec-config-path",
                    type=Path,
                    required=False,
                    help=("Config path for TM-Vec embedding model. "
                          "This is used to encode the proteins as "
                          "vectors to construct the database.")
)
parser.add_argument("--protrans-model",
                    type=Path,
                    default=None,
                    required=False,
                    help=("Model path for the ProTrans embedding model. "
                          "If this is not specified, then the model will "
                          "automatically be downloaded.")
)
parser.add_argument("--device",
                    type=str,
                    default=None,
                    required=False,
                    help=(
                        "The device id to load the model onto. "
                        "This will specify whether or not a GPU "
                        "will be utilized.  If `gpu` is specified "
                        "then the first gpu device will be used."
                    )
)
parser.add_argument("--output",
                    type=Path,
                    required=True,
                    help="Output path for the embedding files"
)

parser.add_argument("--threads",
                    type=int,
                    default=1,
                    required=False,
                    help="Number of threads to use for parallel processing.")

# Load arguments
args = parser.parse_args()

# Set device
if args.device == 'cpu':
    device = torch.device('cpu')
elif torch.cuda.is_available() and args.device is not None:
    if args.device == 'gpu':
        device = torch.device(f'cuda:0')
    else:
        device = torch.device(f'cuda:{int(args.device)}')
else:
    device = torch.device('cpu')

if args.protrans_model is None:
    # Load the ProtTrans model and ProtTrans tokenizer
    tokenizer = T5Tokenizer.from_pretrained("../Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("../Rostlab/prot_t5_xl_uniref50")
else:
    tokenizer = T5Tokenizer.from_pretrained(args.protrans_model, do_lower_case=False )
    model = T5EncoderModel.from_pretrained(args.protrans_model)

#gc.collect()
model = model.to(device)
model = model.eval()
print("ProtTrans model downloaded")
# set up threads
torch.set_num_threads(args.threads)

# Read in query sequences
with FastxFile(args.input_fasta) as query_fasta:
    headers = []
    seqs = []
    for record in query_fasta:
        headers.append(record.name)
        # remove "*"" if present in sequence
        # "*"" denotes stop codon in Prodigal
        seqs.append(record.sequence.replace('*', ''))

flat_seqs = [seqs[i] for i in range(len(seqs))]
print(f"Loaded {len(flat_seqs)} sequences")

# Load the Tm_Vec_Align TM model
tm_vec_model_config = trans_basic_block_Config.from_json(args.tm_vec_config_path)
model_deep = trans_basic_block.load_from_checkpoint(args.tm_vec_model,
                                                    config=tm_vec_model_config,
                                                    map_location=device)
model_deep = model_deep.to(device)
model_deep = model_deep.eval()
print("TM-Vec model loaded")

embeddings = encode(flat_seqs, model_deep, model, tokenizer, device)
os.mkdir(args.output)
path_output = os.path.join(args.output, 'embeddings')
np.save(path_output, embeddings)

print(embeddings.shape)
