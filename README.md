# A Machine Learning Framework for Turbofan Health Estimation via Inverse Problem Formulation
This repository provides the functionalities to reproduce the results provided in a paper under submission in ECML PKDD 2026. 

# How to install
1. Create a virtual environment using conda
```bash
conda create -n virtualenv python=3.10
conda activate virtualenv
```

2. If you want to generate data, you should also install `OpenDeckSMR` package form [here](https://github.com/OpenDeckLab/OpenDeckSMR). 

3. All other requirements could be installed using the requirements file
```bash
pip install -r requirements.txt
```

# How to use
There are different jupyter notebooks provided in this repository to illustrate the use of the package and how to reproduce the results.

- Jupyter Notebook [`01_data_analysis.ipynb`](notebooks/01_data_analysis.ipynb) provides the details concerning the data
- Jupyter Notebook [`02_steady_state.ipynb`](notebooks/02_steady_state.ipynb) provides the details concerning the steady-state models
- Jupyter Notebook [`03_temporal_based.ipynb`](notebooks/03_temporal_based.ipynb) provides the details concerning the temporal-based models


