# BagChain

Source code for the simulations in the paper "[BagChain: A Dual-functional Blockchain Leveraging Bagging-based Distributed Learning](https://arxiv.org/abs/2502.11464)". This project is built upon a previous version of [ChainXim](https://github.com/XinLab-SEU/ChainXim).

## Prerequisites 

### Environment

1. Install Python 3.9 or higher.

2. Install [Graphviz](https://graphviz.org/download/) and add the directory containing the `dot` executable to the PATH environment variable.

3. Install dependencies: `pip install scikit-learn torch matplotlib networkx scipy graphviz pandas numpy urllib3`

### Datasets

Download the datasets from [google drive](https://drive.google.com/drive/folders/1nfQQtPh6Hb9ZyP3CGJeWDKySDU6NwO7b?usp=drive_link) and place them in the `tasks/datasets/` directory.

## Simulation

Run any of the ython scripts starting with 'experiment' to start a simulation. These scripts can generate the simulation results shown in the figures in the simulation section.

| Figure  | Script                                                     |
| ------- | ---------------------------------------------------------- |
| Fig. 7  | experiment_global_base_iid.py                              |
| Fig. 8  | experiment_global_base_niid.py                             |
| Fig. 9  | experiment_imbalance.py                                    |
| Fig. 10 | experiment_sync_random.py                                  |
| Fig. 11 | experiment_network_delay.py (in branch main)               |
| Fig. 12 | experiment_network_delay.py (in branch cross-fork-sharing) |

