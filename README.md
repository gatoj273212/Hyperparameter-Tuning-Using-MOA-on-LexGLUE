# 🚀 MOA-Based Hyperparameter Optimization for Classical Machine Learning on LexGLUE

This repository contains the implementation of Mayfly Optimization Algorithm (MOA) for hyperparameter optimization of classical machine learning classifiers applied to multi-label legal text classification on the LexGLUE ECtHR-B dataset.
The approach combines Legal-BERT embeddings with optimized classical classifiers to demonstrate competitive performance on complex legal NLP tasks.

---

## 📌 Overview

Legal text classification presents challenges such as long documents, severe label imbalance, and domain-specific language.
This project shows that classical machine learning models, when paired with transformer-based embeddings and metaheuristic optimization, can achieve strong results on multi-label legal benchmarks.

The workflow includes:

1. Extracting 768-dimensional embeddings using **Legal-BERT**
2. Training classical classifiers (SVM, Logistic Regression, MLkNN, XGBoost)
3. Using **MOA** for hyperparameter tuning
4. Evaluating models using multi-label metrics such as F1-micro

---

## 🏗️ Architecture

```markdown

Document Text
      ↓
Legal-BERT Embeddings (CLS Vector, 768d)
      ↓
Mayfly Optimization Algorithm (Hyperparameter Search)
      ↓
Optimized Classifiers (SVM, LR, MLkNN, XGB, RF)
      ↓
Evaluation (F1-micro, F1-macro, Hamming Loss)

```
---

## 🔍 Dataset: ECtHR Task B (LexGLUE)

- A multi-label legal judgement prediction dataset  
- Input: Long case descriptions from the European Court of Human Rights  
- Output: Violated ECHR articles (multi-label)  
- Highly imbalanced dataset, making optimization important  

**Dataset Source**
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#ECtHR%20(A)%20(LexGLUE)

### Dataset Availability

The ECtHR-B dataset used in this project is **included in the repository** under the `Dataset/` directory.
The files are provided in compressed `.bz2` format, exactly as released by the official LexGLUE source.

---

## ⚙️ Features

- Legal-BERT embedding pipeline  
- Full Mayfly Optimization Algorithm implementation  
- Classifier-specific hyperparameter encoding  
- Multi-label evaluation metrics  
- Convergence plots for all classifiers  
- Reproducible training and evaluation pipeline  

---

## 📊 Results Summary

Final test performance of optimized models:

| Classifier | F1-Micro | F1-Macro | Hamming Loss |
|------------|----------|----------|---------------|
| **Logistic Regression** | **0.4053** | 0.0322 | 0.00805 |
| **SVM** | 0.3929 | 0.0323 | **0.00614** |
| **MLkNN** | 0.2783 | 0.0169 | 0.01099 |
| **XGBoost** | 0.2061 | 0.0072 | 0.00584 |
| **Random Forest** | 0.2624 | 0.0153 | 0.00980 |

Highlights:

- Logistic Regression achieved the **highest overall score**
- SVM achieved the **lowest Hamming Loss**, giving the most stable performance
- MLkNN improved significantly after MOA tuning
- Tree-based models struggled with dense 768-dimensional embeddings

---

## 🧠 Mayfly Optimization Algorithm — Short Explanation

MOA simulates mayfly behavior to optimize hyperparameters in a continuous search space.

- **Male mayflies** move based on personal best and global best solutions  
- **Female mayflies** move toward the nearest male  
- **Offspring** are generated using linear crossover between male–female pairs  
- **Fitness Function:** F1-micro score computed on the validation split  

MOA effectively balances **exploration** (global search) and **exploitation** (local refinement), making it more powerful than grid search or random search for complex multi-label problems.

---

## 📌 Evaluation Metrics

The following metrics were used to evaluate all classifiers:

- **F1-Micro (Primary Metric for LexGLUE)**  
- **F1-Macro**  
- **Hamming Loss**  
- **Convergence Curves** (performance vs. optimization iterations)  
- **Per-Label Precision, Recall, F1** for detailed analysis

## 📦 Prerequisites

- Python **3.9 or above**
- `pip`
- *(Recommended)* GPU support for faster embedding extraction

### Required Libraries
Install all dependencies using:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run
Each classifier is implemented in a separate Jupyter Notebook, and each notebook independently performs:
- Dataset loading
- Legal-BERT embedding extraction
- Training
- Validation
- Testing
- Evaluation metrics computation

### 1. Clone the Repository
```bash
git clone https://github.com/gatoj273212/Hyperparameter-Tuning-Using-MOA-on-LexGLUE.git
cd Hyperparameter-Tuning-Using-MOA-on-LexGLUE
```
### 2. Run Experiments

Open Jupyter Notebook:
```bash
jupyter notebook
```
Then run any of the following notebooks depending on the classifier:

Logistic Regression
```bash
Lexglue_LR_new.ipynb
```

Support Vector Machine (SVM)
```bash
Lexglue_SVM.ipynb
```

MLkNN
```bash
Lexglue_MLKNN.ipynb
```

Random Forest
```bash
Lexglue_Randomforest.ipynb
```

XGBoost
```bash
Lexglue_XGBoost_new.ipynb
```
Each notebook is self-contained and can be executed independently from top to bottom.

## 📘 References

- Chalkidis, I., Fergadiotis, M., Tsarapatsanis, D., Aletras, N., & Androutsopoulos, I.  
  **LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.**  
  arXiv:2110.00976 (2021).  
  https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html#ECtHR%20(A)%20(LexGLUE)

- Zervoudakis, K., & Tsafarakis, S.  
  **Mayfly Optimization Algorithm.**  
  Computers & Industrial Engineering, Elsevier.
  
- HuggingFace Model Card  
  **nlpaueb/legal-bert-base-uncased**  
  https://huggingface.co/nlpaueb/legal-bert-base-uncased



