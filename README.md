# 🔬 AEPMA: Peptide-Microbe Association Prediction via Autoevolutionary Heterogeneous Graph Learning

![release](https://img.shields.io/badge/release-v1.0-blue)
![open issues](https://img.shields.io/badge/open%20issues-0-brightgreen)
![pull requests](https://img.shields.io/badge/pull%20requests-0%20open-brightgreen)

![AEPMA Framework](https://github.com/ahu-bioinf-lab/AEPMA/blob/master/AEPMA.png)

---

## 🧠 Overview

**AEPMA** is a computational framework for predicting potential associations between antimicrobial peptides (AMPs) and microbes.  
It constructs a novel **peptide-microbe-disease heterogeneous graph** and applies an **autoevolutionary information aggregation model** to capture complex biological interactions — enabling efficient AMP discovery and repurposing.

---

## 🚀 Key Features

- 📌 Builds a peptide-microbe-disease heterogeneous graph  
- 🧬 Leverages an **autoevolutionary aggregation** model  
- 📊 Captures semantic, structural, and spatiotemporal dependencies  
- 🧪 Offers a cost-effective alternative to biological experiments

---

## ⚙️ Installation

```bash
pip install -r requirements.txt

## 📘 Usage
## Step 1: Data Preprocessing
Run the `preprocess.py` script to prepare the input heterogeneous network. This step ensures your data is correctly processed for subsequent training and prediction.
## Step 2: Autoevolutionary heterogeneous graph
Execute the `train_search.py` script to identify the optimal adaptive meta-graph for DTI. This stage involves a search process to determine the meta-graph structure that best suits DTI prediction.
## Step 3: Prediction
Use the `train.py` script to apply the adaptive meta-graph to DTI prediction. This step employs the best adaptive meta-graph from the previous step to make predictions and generate results.
Following these steps in order will help ensure successful replication of the results presented in our manuscript. If you encounter any challenges during execution or need more detailed information, please consult our code documentation and program instructions for guidance on parameter settings and data preparation.

## 📬 Contact

For questions or suggestions, please contact [d202481545@hust.edu.cn] or open an issue on GitHub.
