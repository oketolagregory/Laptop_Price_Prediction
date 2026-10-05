**Laptop Price Prediction**


A regression pipeline predicting laptop prices from technical specifications, built for MSc Data Science coursework at Bournemouth University.


**Overview**


Laptop pricing depends on a complex mix of specs (brand, CPU, RAM, GPU, screen, etc.) that interact non-linearly. This project builds and compares multiple regression approaches to predict price from a dataset of 1,303 laptops across 11 raw features, with a focus on rigorous feature engineering and model comparison rather than settling for a single off-the-shelf approach.


**Method**


**Exploratory analysis**


Univariate and multivariate analysis across all features

Scatter plots of each feature against price to identify relationships worth engineering


**Feature engineering**


Parsed noisy compound fields into structured features: CPU model + clock speed, GPU model, memory type + size (handling GB/TB unit conversion), resolution into pixel density (PPI) plus Touchscreen/IPS flags

Grouped rare brand categories into an "Others" bin to reduce sparsity

Binned screen sizes to reduce dimensionality while preserving signal

One-hot encoded categorical variables (Brand, Type, OS)



**Feature selection (benchmarked three approaches)**


Recursive Feature Elimination (RFE): best performance at 15-20 features

Principal Component Analysis (PCA): first 4 components explained ~99% of variance, but didn't improve R2 over the full feature set

Correlation analysis: selected top correlated features, then dropped colinear pairs (HDD/SSD, Gaming/Nvidia_Gpu)


Correlation-based selection was used going forward, as it gave the clearest, most interpretable feature set without sacrificing performance.


Modelling Benchmarked five approaches via cross-validation, then hyperparameter-tuned the top performers with GridSearchCV:


XGBoost Regressor

LightGBM Regressor

Random Forest Regressor

Voting Regressor (ensemble of the above three)

Neural Network (Keras, with Dropout layers and EarlyStopping to control overfitting)



**Evaluation**

R2 and MAE on held-out test data

Residual plots and prediction error plots (via yellowbrick) for each model to diagnose fit quality beyond a single metric


**Results**

Model	R2	MAE

Voting Regressor (ensemble)	0.871	0.041

XGBoost	0.861	0.164

LightGBM	0.858	0.168

Random Forest	0.847	0.174

Neural Network	0.813	0.198


The Voting Regressor, combining XGBoost, LightGBM, and Random Forest, outperformed every individual model on both R2 and MAE, showing that ensembling correlated-but-distinct tree-based learners captured patterns no single model fully captured on its own.


**Tech Stack**

Python · pandas · scikit-learn · XGBoost · LightGBM · Keras/TensorFlow · yellowbrick (model diagnostics) · seaborn / matplotlib


**Files**
Laptop_Price_Prediction.ipynb — full pipeline: EDA, feature engineering, feature selection benchmarking, model training, evaluation


**Key Learnings**
Feature selection method choice matters: correlation-based selection outperformed PCA here, since PCA's components weren't as interpretable or performant for this dataset
Ensembling diverse but individually strong models (here, three different tree-based algorithms) can meaningfully outperform any single model, even a well-tuned one
