# 🧠 Cold-Start Fraud Detection System  
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> **IBM Hackathon — Finance Track 2025**  
> A transparent, explainable Machine Learning system for detecting **fraudulent financial transactions**, designed to **generalize to new clients (cold-start problem)** using a robust Logistic Regression pipeline.

---

## 🎯 Objective

Financial institutions face an evolving enemy: **fraudsters change behavior faster than traditional systems can adapt.**

This project aims to:
- 🔍 **Predict** whether a transaction is fraudulent (`1`) or legitimate (`0`)
- ⚙️ **Generalize** to *unseen clients* (cold start)
- 📈 **Maximize precision** at the top of the analyst queue  
- ✅ **Ensure transparency** and explainability for compliance  

---

## 💡 Concept & Story

> “You can take the blue pill, trust your legacy security systems…  
> or take the red pill — and see how deep the rabbit hole goes.”  

We chose the **red pill** — to **see the code** behind the illusion of security.

Our system empowers banks to move from **reactive defense** to **proactive detection**, identifying *when, how, and who* commits fraud — *before* the losses occur.

---

## 🧩 System Architecture

```text
Data Ingestion → Feature Engineering → Model Training
                     ↓
          Temporal + Group Validation
                     ↓
     Threshold Optimization & Evaluation
                     ↓
       Dashboard Visualization & Insights

### Core Design Choices
- 🕐 **Temporal split** → simulate future behavior  
- 👥 **GroupKFold by client_id** → handle cold-start clients  
- ⚖️ **Class imbalance** handled via `class_weight='balanced'`  
- 🧮 **Metric optimization**: PR-AUC, Precision@k, F1-max threshold  
- 💾 **Artifacts** saved via `joblib` for reproducibility  

---

## 🧠 Model Details

| Component | Description |
|------------|--------------|
| **Algorithm** | Logistic Regression (`saga`, `class_weight="balanced"`) |
| **Preprocessing** | `StandardScaler` + `OneHotEncoder` |
| **Validation** | Temporal holdout + GroupKFold (client/card) |
| **Metric Focus** | PR-AUC, ROC-AUC, Precision@k |
| **Threshold** | F1-max on validation (~0.99996) |

---

## 📊 Results

| Metric | Score |
|--------|--------|
| **ROC-AUC** | ≈ 0.98 |
| **PR-AUC** | ≈ 0.47 |
| **Precision@1% reviewed** | ≈ 17% |
| **Recall@1% reviewed** | ≈ 69% |

💡 *With only 1% of transactions reviewed, ~69% of actual frauds are detected.*

---

## 🖥️ Fraud Intelligence Dashboard

Our **Fraud Intelligence Platform** turns data into decisions:  

- 🕒 **WHEN** → Fraud spikes between **2 PM–4 PM** → smarter staffing  
- 🌐 **HOW** → **Online transactions** dominate fraud activity  
- 👤 **WHO** → Ranked **Top-20 high-risk transactions** for instant review  
- 📉 **WHY** → Clear score distribution + threshold visualization  

> This isn’t just reporting — it’s an **action console**.

![Dashboard](fraud_detection_dashboard.jpeg)

---

## 🔧 Tech Stack

| Category | Tools |
|-----------|--------|
| **Language** | Python 3.10+ |
| **Libraries** | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib` |
| **Environment** | Jupyter Notebook |
| **Visualization** | Custom fraud analysis dashboard |

---

## 🧭 Future Work

- ⚡ **Advanced Models:** LightGBM / XGBoost for non-linear patterns  
- 🔗 **Graph Features:** card ↔ merchant ↔ device relations  
- 🧰 **Deployment:** FastAPI + Kafka for near-real-time scoring  
- 📊 **Monitoring:** Drift detection (PSI), calibration, and A/B threshold testing  
- 💬 **Human-in-the-loop:** Analyst feedback integration  

---

## 👥 Team

**Data & AI Engineering Students — IBM Hackathon 2025**

- 🧑‍💻 **Rayan Hamadeh** — Project Lead & ML Engineer  
- 👥 *Collaborators:* [Add your teammates here]

---

## 📂 Repository Structure

```bash
├── Finance.ipynb                  # Main notebook (EDA + pipeline + results)
├── fraud_detection_dashboard.jpeg # Dashboard visualization
├── instructions.pdf               # IBM Hackathon challenge brief
├── outputs/
│   ├── model_meta.joblib          # Trained model
│   └── submission.csv             # Evaluation predictions
└── README.md                      # Project documentation
