# BERT Active Learning Pipeline

The purpose of the active learning pipeline is to spin up performant machine learning models using as little data as possible.

## Installation Instructions

### Mac OS (Mac M1 Chip)
1. Download and install a conda- like package management system. In these instructions, I use Mamba because it resolves dependencies quickly, but if you already have conda installed, you can use that too. Just replace the word `mamba` with `conda` in the below instructions.

2. Set up the environment
```
CONDA_SUBDIR=osx-arm64 mamba create -n active_learning python=3.9 -c conda-forge
mamba activate active_learning
mamba env config vars set CONDA_SUBDIR=osx-arm64 // For changes to take effect, you need to deactivate/ reactivate (next 2 lines)
mamba deactivate
mamba activate active_learning
```

3. Install tensorflow
```
mamba install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

4. Install rust as a prerequisite for the transformers package
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

5. Finally, install the rest of the packages using pip
```
pip install -r requirements_m1.txt
```