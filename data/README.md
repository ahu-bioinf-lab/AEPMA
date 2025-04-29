To make the code more usable for others and provide flexibility in adapting it to different datasets, we can organize the GitHub documentation and code explanation into the following sections:

---

# **Data Preprocessing for Microbe-Peptide-Disease Interaction Networks**

## **Overview**

This repository provides a data preprocessing pipeline for building interaction networks in the context of microbe-peptide-disease association prediction. The preprocessing is designed to work with several data files, such as `pep-microbe.dat`, `pep-pep.dat`, `microbe-microbe.dat`, `microbe-disease.dat`, and `disease-disease.dat`, which contain associations between peptides, microbes, and diseases, along with their respective ratings or weights.

The processed data is saved as adjacency matrices and positive/negative pairs for downstream machine learning tasks. The code is designed to be flexible, enabling users to adapt the preprocessing pipeline to their own datasets.

## **Features**
- Prepares interaction matrices between peptides, microbes, and diseases.
- Generates positive and negative pairs for peptide-microbe interactions.
- Provides adjacency matrices that represent connections between nodes (peptides, microbes, and diseases) in the network.
- Supports dynamic offsets for flexible network structure configurations.

---

## **Getting Started**

### **1. Clone the Repository**

Start by cloning the repository:

```bash
git clone https://github.com/yourusername/interaction_network_preprocessing.git
cd interaction_network_preprocessing
```

### **2. Prerequisites**

You need to have the following libraries installed to run the preprocessing script:
- `numpy`
- `pandas`
- `scipy`
- `pickle`

You can install the necessary packages using `pip`:

```bash
pip install numpy pandas scipy
```

### **3. Input Files**

The following input data files are required for preprocessing:

1. **pep-microbe.dat**: Peptide-Microbe interactions (peptide ID, microbe ID, rating).
2. **pep-pep.dat**: Peptide-Peptide interactions (peptide 1 ID, peptide 2 ID, weight).
3. **microbe-microbe.dat**: Microbe-Microbe interactions (microbe 1 ID, microbe 2 ID, weight).
4. **microbe-disease.dat**: Microbe-Disease associations (microbe ID, disease ID, rating).
5. **disease-disease.dat**: Disease-Disease interactions (disease 1 ID, disease 2 ID, weight).

These files should be placed in a directory structure like this:

```plaintext
data/
  pep-microbe.dat
  pep-pep.dat
  microbe-microbe.dat
  microbe-disease.dat
  disease-disease.dat
```

Make sure these files are in CSV format, with appropriate columns as described above.

---

## **Script Overview**

### **1. `pretreatment_Pep(prefix)` Function**

This function processes the interaction datasets and generates:
- Positive and negative pairs for peptide-microbe interactions.
- Adjacency matrices for peptide, microbe, and disease relationships.
- A node type array to differentiate between peptides, microbes, and diseases.
- Saves all preprocessed data into `.npy` and `.pkl` files for easy use in subsequent tasks.

### **Adjustable Parameters**
- `prefix`: Directory path containing the input data files (`pep-microbe.dat`, `pep-pep.dat`, etc.).
- `offsets`: A dictionary that controls the offset values for peptide, microbe, and disease IDs in the adjacency matrices.

### **Key Parameters Explained**
- `pm`: Peptide-Microbe interaction matrix.
- `pp`: Peptide-Peptide interaction matrix.
- `mm`: Microbe-Microbe interaction matrix.
- `md`: Microbe-Disease association matrix.
- `dd`: Disease-Disease interaction matrix.
- `node_types`: A numpy array that differentiates between peptides (type 0), microbes (type 1), and diseases (type 2).
- `adjs_offset`: A dictionary of sparse adjacency matrices for each interaction type.

### **Example Usage**

```python
# Example of running the preprocessing script on your data

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle

# Path to your dataset directory
prefix = "path/to/your/data"

# Run the pretreatment function
pretreatment_Pep(prefix)
```

After running the script, the following output files will be generated:
- **node_types.npy**: Numpy file containing node types (peptides, microbes, diseases).
- **neg_pairs_offset.npz**: Numpy file containing negative peptide-microbe pairs.
- **pos_pairs_offset.npz**: Numpy file containing positive peptide-microbe pairs.
- **combined_matrices.npz**: Numpy file containing combined adjacency matrices for peptide-peptide, microbe-microbe, and disease-disease relationships.
- **adjs_offset.pkl**: Pickle file containing sparse adjacency matrices for various interactions.

---

## **Customizing for Your Dataset**

To adapt this code to your own dataset, you can make the following adjustments:

1. **Data Format**: Ensure that your input files follow the format described in the "Input Files" section above. If your data is in a different format, you will need to modify the CSV reading process to correctly parse your files.

2. **Offsets and Node Types**: If you have more types of nodes (e.g., additional categories of entities), you can extend the `offsets` dictionary and modify the `node_types` array accordingly.

3. **Interaction Matrices**: If you have additional interaction types, you can add new adjacency matrices to the `adjs_offset` dictionary. Just ensure that the `node_types` array accommodates the new nodes.

---

## **Advanced Options and Parameters**

The script is designed to be flexible for various types of networks. Here are some key adjustments you can make:

- **Adjusting the Number of Negative Pairs**: The number of negative pairs is currently set to be 1x the number of positive pairs. If you want to use a different ratio, modify the following line:

```python
indices_neg = indices_neg[:pm_pos.shape[0] * 1]
```

Change `1` to another value to control the negative pair ratio.

- **Custom Node Offsets**: The current offset values for peptides, microbes, and diseases are:

```python
offsets = {'p': 4050, 'm': 4050 + 131, 'd': 4050 + 131 + 161}
```

If your dataset has different numbers of peptides, microbes, or diseases, adjust these values accordingly.
