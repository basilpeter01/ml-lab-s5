# Machine Learning Laboratory (Semester 5)

This repository contains lab exercises and implementations for the Semester 5 Machine Learning Lab coursework. The experiments cover supervised learning algorithms, gradient descent, normal equation, MLE, and MAP, regression variants, and classification models implemented in Python using Jupyter Notebooks.

---

## Lab Experiments & Notebooks

| # | Topic / Experiment | Description | Dataset | Notebook Link |
|---|---|---|---|---|
| 01 | Simple & Multiple Linear Regression | Implementation of linear regression using `scikit-learn`, evaluating goodness of fit with Mean Squared Error (MSE) and R² score, along with residual analysis. | California Housing | [linear-regression.ipynb](./linear-regression.ipynb) |
| 02 | Linear Regression from Scratch | Analytical solution via the Normal Equation $(X^T X)^{-1} X^T y$ and iterative optimization via Batch Gradient Descent, with fitted regression line plotting. | California Housing | [regression-california-housing.ipynb](./regression-california-housing.ipynb) |
| 03 | Polynomial Regression | Modeling non-linear relationships using polynomial feature transformations of varying degrees to evaluate variance and fit. | Auto MPG | [polynomial-regression-with-mpg.ipynb](./polynomial-regression-with-mpg.ipynb) |
| 04 | Linear vs. Polynomial Comparison | Comparative evaluation of linear versus polynomial regression models across performance metrics (MSE, R²). | Auto MPG | [mpg-regression-comparison.ipynb](./mpg-regression-comparison.ipynb) |
| 05 | Ridge & Lasso Regularization | Implementation of L2 (Ridge) and L1 (Lasso) penalty terms with `GridSearchCV` hyperparameter tuning to inspect shrinkage, feature selection, and test performance. | Diabetes Dataset | [ridge-lasso-regression.ipynb](./ridge-lasso-regression.ipynb) |
| 06 | Logistic Regression | Binary classification pipeline including feature scaling (`StandardScaler`) and metric computation (Accuracy, Precision, Recall, and F1-Score). | Pima Indians Diabetes | [logistic-regresssion.ipynb](./logistic-regresssion.ipynb) |
| 07 | Parameter Estimation: MLE vs. MAP | Comparison between unregularized Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation using L1 (Laplace prior) and L2 (Gaussian prior) regularizers, analyzing weight sparsity and decision accuracy. | Breast Cancer Wisconsin | [mle-map-logistic.ipynb](./mle-map-logistic.ipynb) |
| 08 | Naive Bayes Text Classification | Document classification using Bag-of-Words representations (`CountVectorizer`) with Multinomial and Bernoulli Naive Bayes models. | 20 Newsgroups | [naive-bayes.ipynb](./naive-bayes.ipynb) |
| 09 | k-Nearest Neighbors (k-NN) | Multi-class image classification using the k-NN algorithm, exploring accuracy across distance metrics and varying values of $k$. | Fashion-MNIST | [knn.ipynb](./knn.ipynb) |
| 10 | Decision Tree Classification | Supervised classification using `DecisionTreeClassifier` with data cleaning, tree visualization (`plot_tree`), and test set accuracy evaluation. | Online Retail Dataset | [decision-tree.ipynb](./decision-tree.ipynb) |

---

## Repository Structure

```text
ml-lab-s5/
├── decision-tree.ipynb                 # Lab: Decision Tree classifier and visualization
├── knn.ipynb                           # Lab: k-Nearest Neighbors on Fashion-MNIST
├── linear-regression.ipynb             # Lab: Scikit-learn Linear Regression
├── logistic-regresssion.ipynb          # Lab: Logistic Regression with evaluation metrics
├── mle-map-logistic.ipynb              # Lab: MLE vs MAP estimation with regularizers
├── mpg-regression-comparison.ipynb     # Lab: Comparative study of regression models
├── naive-bayes.ipynb                   # Lab: Text classification via Naive Bayes
├── polynomial-regression-with-mpg.ipynb# Lab: Polynomial Regression modeling
├── regression-california-housing.ipynb # Lab: Linear regression from scratch (GD & Normal Eq)
├── ridge-lasso-regression.ipynb        # Lab: Ridge (L2) and Lasso (L1) regularization
└── README.md                           # Lab coursework documentation
```

---

## Core Topics Covered

1. **Regression Analysis**
   - Ordinary Least Squares (OLS) closed-form solution (Normal Equation)
   - Iterative parameter updates using Batch Gradient Descent
   - Non-linear modeling using Polynomial Feature transformations
   - Regularization techniques: Ridge ($L_2$ shrinkage) and Lasso ($L_1$ sparsity) with cross-validation

2. **Probabilistic & Statistical Modeling**
   - Maximum Likelihood Estimation (MLE)
   - Maximum A Posteriori (MAP) estimation under Gaussian and Laplace priors
   - Generative classification using Multinomial and Bernoulli Naive Bayes with tokenized text

3. **Instance-Based & Tree-Based Learning**
   - Distance-based prediction with $k$-Nearest Neighbors ($k$-NN)
   - Partition-based classification using Decision Trees with impurity metrics (Gini / Entropy)

4. **Model Evaluation & Diagnostics**
   - Regression metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$)
   - Classification metrics: Confusion matrices, Accuracy, Precision, Recall, and Macro/Weighted $F_1$-Scores
   - Residual plots and decision boundaries

---

## Setup & Execution

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/basilpeter01/ml-lab-s5.git
cd ml-lab-s5
```

Install standard scientific computing libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Running the Notebooks

Launch Jupyter Notebook within the repository folder:

```bash
jupyter notebook
```

Select any experiment from the file browser and run the cells sequentially to reproduce outputs, plots, and evaluation tables.
