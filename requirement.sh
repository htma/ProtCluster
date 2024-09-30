
# First create a conda environment with python=3.12 installed.

conda create -n tmvec faiss-cpu python=3.12 -c pytorch
conda activate tmvec

# for GPU, if fails, install mkl via `conda install mkl=2021 mkl_fft`
# module required environment
module load gcc/11.2 anaconda/2024.02-1 cuda/12.1

# Create a new conda enviroment with pytorch for gpu and download faiss-gpu
conda create -n tmvec faiss-gpu python=3.12 -c pytorch

# Initializing a Conda environment
conda init

# Update .bashrc
source ~/.bashrc

# Activate the tmvec environment
conda activate tmvec

# Add the conda-forge to the channels
conda config --prepend channels conda-forge
conda config --add channels bioconda

# Install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install pandas -c conda-forge

pip install pytorch_lightning
conda install transformers -c conda-forge
conda install pysam -c bioconda
pip install SentencePiece
conda install -c bioconda biopython
conda install -c conda-forge tqdm


