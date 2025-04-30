## Flexible Network Input Guide for Drug-Microbe Data

This guide will help users adjust the network input format, understand the parameter descriptions, and use the preprocessing and matrix generation code to process various datasets.

### 1. **Adjustable Network Input Format**

In the `Preprocess.ipynb` script, the input files follow the format below, but you can modify them as needed:

- **Drug-Microbe Relation (`adj.dat`)**:
    - Format: `drug_id, microbe_id, rating`
    - `drug_id`: Drug ID (1-based indexing)
    - `microbe_id`: Microbe ID (1-based indexing)
    - `rating`: Interaction rating (binary or continuous)

- **Drug-Drug Similarity (`drugsimilarity.dat`)**:
    - Format: `drug1_id, drug2_id, weight`
    - `drug1_id, drug2_id`: Drug pair IDs (1-based indexing)
    - `weight`: Similarity score (continuous)

- **Microbe-Microbe Similarity (`microbesimilarity.dat`)**:
    - Format: `microbe1_id, microbe2_id, weight`
    - `microbe1_id, microbe2_id`: Microbe pair IDs (1-based indexing)
    - `weight`: Similarity score (continuous)

**Notes**:
- Ensure that the input file format matches the one used in the script's `pd.read_csv()` part. If the file uses a different delimiter (e.g., tab-separated), modify the `delimiter` parameter accordingly.
- If using a different dataset, adjust the column names in the script as needed.

### 2. **Parameter Descriptions**

In the `Preprocess.ipynb` script, the following are important parameters and their descriptions:

- **`prefix`**: The directory path where the input files are stored.
    - Set this to the folder containing the `adj.dat`, `drugsimilarity.dat`, and `microbesimilarity.dat` files.

- **`node_types`**: An array representing the types of nodes. `0` represents drug nodes, and `1` represents microbe nodes. This array is used to distinguish between different node types in the graph.

- **Adjacency Matrices**:
    - Different adjacency matrices represent various relationships:
        - `Drug-Microbe`: Encodes the interaction between drugs and microbes.
        - `Drug-Drug`: Encodes the similarity between drugs.
        - `Microbe-Microbe`: Encodes the similarity between microbes.
    - These matrices are stored as sparse matrices (`scipy.sparse.coo_matrix`) to save memory.

- **`offsets`**: Used to distinguish between drug and microbe node types with offset values. Ensure that the drug ID (`max_drug_id`) and microbe ID (`max_microbe_id`) are adjusted dynamically after computation.

- **Data Format Conversion**:
    - The input data uses 1-based indexing, which the code will convert to 0-based indexing for further processing.
    - Make sure to adjust the script according to the indexing type of your dataset.

### 3. **Documentation**

The `Preprocess.ipynb` script already contains all necessary steps and functional modules, including data loading, preprocessing, matrix generation, and saving results. Users only need to adapt the file paths and column names based on their dataset.

### 4. **How to Load and Use Processed Data**

The processed data will be saved in `.npz` and `.pkl` files for easy loading:
- **Processed Matrices**:
    - All matrices (e.g., `dp_matrix`, `pd_matrix`, etc.) will be saved in `.npz` files.
    - Adjacency matrices will be saved in serialized `.pkl` files.
  
**In the `Preprocess.ipynb` script, you can see how to load these saved files and use them for subsequent network construction or model training.**

---

This documentation points to the contents and operations already included in the `Preprocess.ipynb` script, ensuring that other users can understand how to adjust and use the script to process different drug, microbe, and disease datasets.
