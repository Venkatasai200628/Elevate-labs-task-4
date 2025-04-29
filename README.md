# Elevate-labs-task-4
This project is part of the AI & ML Internship task to build a **binary classifier** using **Logistic Regression**. The goal is to classify whether a tumor is benign or malignant based on medical imaging features.

Dataset
name Breast Cancer Wisconsin Diagnostic Dataset
Features: 30 numeric features (e.g., radius, texture, concavity, etc.)
Target: 
  M = Malignant (1)
  B = Benign (0)

Tools & Libraries
Python 3.x
Pandas
Scikit-learn
Matplotlib
Seaborn

Steps Performed

1.Data Preprocessing
Dropped irrelevant columns (id, Unnamed: 32)
Encoded target labels: M → 1, B → 0
Handled scaling using StandardScaler

2.Train/Test Split
80% training, 20% testing using train_test_split

3.Model Building
   Used LogisticRegression from sklearn.linear_model
   Trained on standardized features

4. Evaluation
   confusion Matrix
   Precision, Recall, F1-score
   ROC Curve & AUC Score

Results

         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
2  84300903         M        19.69  ...          0.3613                  0.08758          NaN
3  84348301         M        11.42  ...          0.6638                  0.17300          NaN
4  84358402         M        20.29  ...          0.2364                  0.07678          NaN

         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
2  84300903         M        19.69  ...          0.3613                  0.08758          NaN
3  84348301         M        11.42  ...          0.6638                  0.17300          NaN
4  84358402         M        20.29  ...          0.2364                  0.07678          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
2  84300903         M        19.69  ...          0.3613                  0.08758          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
2  84300903         M        19.69  ...          0.3613                  0.08758          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
         id diagnosis  radius_mean  ...  symmetry_worst  fractal_dimension_worst  Unnamed: 32
0    842302         M        17.99  ...          0.4601                  0.11890          NaN
1    842517         M        20.57  ...          0.2750                  0.08902          NaN
2  84300903         M        19.69  ...          0.3613                  0.08758          NaN
3  84348301         M        11.42  ...          0.6638                  0.17300          NaN
4  84358402         M        20.29  ...          0.2364                  0.07678          NaN

[5 rows x 33 columns]



Visualizations
Confusion Matrix Heatmap
ROC Curve Plot

What I Learned
Logistic regression fundamentals
Sigmoid function and its role in binary classification
Difference between precision, recall, and AUC
Threshold tuning and evaluation trade-offs


Author
Nalluri Venkata Sai

