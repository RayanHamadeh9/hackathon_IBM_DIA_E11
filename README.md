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
