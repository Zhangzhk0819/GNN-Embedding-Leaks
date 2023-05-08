Official implementation of ["Inference Attacks Against Graph Neural Networks" (USENIX Security 2022)](https://www.usenix.org/conference/usenixsecurity22/presentation/zhang-zhikun)

# Overview of the Code

The code entry is main.py, it will invoke the experimental classes in 'exp/' folder to conduct different experiments. For example, if you run the code from command line
```
python main.py --attack 'graph_reconstruction'
```
Notice that the arguments are optional, you can specify the default values in the main.py file.

# Code Structure

* main.py: The entry of the whole project, including the configurations of the logger, the parameters used in the experiments.
* config.py: Define some constants used in the experiments, including various data paths.
* exp/: This folder contains the main procedure of different experiments, where each experiment correspond to one class. They will invoke functions implemented in other 'lib_' folders.
* lib_dataset/: This folder contains the implementations for data pre-precessing. We will store the preprocessed data in pickle format to speed up the main experiments.
* lib_gnn_model/: This folder contains different gnn models.
* lib_plot/: This folder contains code for plotting
* temp_data/: This folder is used to store the raw data, the processed data and some temporary data. I do not include this folder in the github, you can create the required folders following the path specified in config.py.

# Directory Structure under `temp_data`
```
.
├── attack_data
│	├── graph_reconstruct
│	├── property_infer
│	├── property_infer_basic
│	└── subgraph_infer
├── defense_data
│	├── property_infer
│	└── subgraph_infer
├── gae_model
│	├── AIDS_diff_pool
│	├── AIDS_mean_pool
│	├── AIDS_mincut_pool
│	└── fine_tune
├── model
│	├── model_AIDS
│	├── para_AIDS
├── original_dataset
│	├── AIDS
├── split
│	└── 20
└── target_model
    └── 20
```

# Citation

```
@inproceedings{zhang22usenix,
author = {Zhikun Zhang and Min Chen and Michael Backes and Yun Shen and Yang Zhang},
title = {{Inference Attacks Against Graph Neural Networks}},
booktitle = {USENIX Security Symposium (USENIX Security)},
year = {2022}
}
```
