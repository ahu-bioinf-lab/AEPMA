# 🔬 AEPMA: Peptide-Microbe Association Prediction via Autoevolutionary Heterogeneous Graph Learning

![release](https://img.shields.io/badge/release-v1.0-blue)
![open issues](https://img.shields.io/badge/open%20issues-0-brightgreen)
![pull requests](https://img.shields.io/badge/pull%20requests-0%20open-brightgreen)

![AEPMA Framework](https://github.com/ahu-bioinf-lab/AEPMA/blob/master/AEPMA.png)

---

## 🧠 Overview
[AEPMA PDF](https://watermark.silverchair.com/bbaf334.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA00wggNJBgkqhkiG9w0BBwagggM6MIIDNgIBADCCAy8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM64O0zWgvw2MWTpTOAgEQgIIDAOX9OZdyFyQ2t7hJxrsSvaM3g0uEmFOP8aGu60dxhgUMazmd8jrmK-WrVbxf_-bFaDf6P06uulgXvJ6BL6dwKWzpWuWe-LaAeqi-PvuM8Ycqk-S7-0uM-__MPwTy_L0gWjPZd_O52Yrsq-sFr4VRxa-fbB48LTcHwVygC7t8SEDXMnxVXCSgF_isspUKiHrYDCOPXgvdg_UwBT8Tku3Gm6BWPjsWapr6YU2BQLiqZRE-YaCKtii8mLFx79MHqZxHa_CAWYpO-AujTaPCYN1Z7HYY83NMkRPpYWFl8Q5eHhIcGwAi-xgaVw-bqQEjvRT7-gU-NRtHD0PvhwyzN4TVJR6JwP1mRMZypL-NhaTrCJed35lL_XNbmFwIYt0pbv9HcIZ0rHonPiIT3gS6RnyHXZlMJauMi9kgfGbK_yVAFGicM1eu9rT9tXMbkrFoeifga7K0g7YcDlQw0Kmjy0yzXFvNhO8ag3F-rlS3zzFPkm-G8YgGous_ia9iYC6pG_BwP5xUJZnIqpAVaFUrOs0gvyqfpg69P4cz1DL-SqnaNxAi5rJ3GuBu4AEvFc6lwY3Ef5vPnn6rvvd7zHmdZZSCRR_N58vZLeD_9vMZPbkmNb_WS1a7aQs2APAZ1CejBjSzbddPZhJRFR1tv8CAKWht4FPe8K3VtJoFUpdUg1sp7Y8FuvPqKn5rgYvfVusVRG3-IhMbUPQnA-55Fb_nud7J6XgrL71xDnUldGIYuPKvq3zUASMvZAFzremH2g9jV-dQ3FpT2ApO-IjqOqbz2TyxxqUHNc3TrQ7wD7TMDq1LQ7yCOxfPfJ725V4OD2gZs8KP1xID5iXFyMFKtqN4M0NKuaz_Scoe5YS5m00ZhMqXUBWpixLdJ2EsboUKk8bWufs8EV1-Ir20Ebwj-zelDv1Slz9zdkG_qBBh_O7mZtK98HeYkQlhz1RtfylmcpDLX2pito7P7xY7jo9D-w39KIbRFmMWFg_O-ZY5lmOdleoipsiz5NPM5xgnmCeshupdiHtwPA)

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
```

---

## 📘 Usage

## Step 1: Data Preprocessing
Run the `preprocess.py` script to prepare the input heterogeneous network. This step ensures your data is correctly processed for subsequent training and prediction.
## Step 2: Autoevolutionary heterogeneous graph
Execute the `train_search.py` script to identify the optimal adaptive meta-graph for DTI. This stage involves a search process to determine the meta-graph structure that best suits DTI prediction.
## Step 3: Prediction
Use the `train.py` script to apply the adaptive meta-graph to DTI prediction. This step employs the best adaptive meta-graph from the previous step to make predictions and generate results.
Following these steps in order will help ensure successful replication of the results presented in our manuscript. If you encounter any challenges during execution or need more detailed information, please consult our code documentation and program instructions for guidance on parameter settings and data preparation.

---

## 📬 Contact

For questions or suggestions, please contact [d202481545@hust.edu.cn] or open an issue on GitHub.
