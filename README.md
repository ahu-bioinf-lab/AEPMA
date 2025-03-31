# AEPMA: Peptide-Microbe association prediction based on autoevolutionary heterogeneous graph learning
![AEPMA Framework](https://github.com/ahu-bioinf-lab/AEPMA/blob/master/AEPMA.png)
## Overview

AEPMA is a computational framework for predicting potential associations between antimicrobial peptides (AMPs) and microbes. By constructing a novel peptide-microbe-disease network and employing an autoevolutionary information aggregation model, AEPMA can effectively capture the complex relationships in heterogeneous biological networks for AMP discovery and repurposing.

## Key Features

- Constructs a comprehensive peptide-microbe-disease heterogeneous network
- Implements an autoevolutionary information aggregation model for representation learning
- Automatically aggregates semantic information from heterogeneous networks
- Incorporates spatiotemporal dependencies and heterogeneous interactions
- Provides an efficient computational alternative to costly biological experiments

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
## Step 1: Data Preprocessing
Run the `preprocess.py` script to prepare the input heterogeneous network. This step ensures your data is correctly processed for subsequent training and prediction.
## Step 2: Autoevolutionary heterogeneous graph
Execute the `train_search.py` script to identify the optimal adaptive meta-graph for DTI. This stage involves a search process to determine the meta-graph structure that best suits DTI prediction.
## Step 3: Prediction
Use the `train.py` script to apply the adaptive meta-graph to DTI prediction. This step employs the best adaptive meta-graph from the previous step to make predictions and generate results.
Following these steps in order will help ensure successful replication of the results presented in our manuscript. If you encounter any challenges during execution or need more detailed information, please consult our code documentation and program instructions for guidance on parameter settings and data preparation.

## Contact

For questions or suggestions, please contact [d202481545@hust.edu.cn] or open an issue on GitHub.
