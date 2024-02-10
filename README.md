# Heart Failure Prediction Using ANN
Welcome to the Heart Failure Prediction project! This project focuses on utilizing Artificial Neural Networks (ANN) to predict heart failure based on various health-related features. It serves as a valuable resource for students and researchers interested in exploring predictive modeling for healthcare applications.
Here, provide a brief introduction to the project, explaining the motivation behind predicting heart failure, the significance of using ANN for this task, and the goals of the project.

## Table of Contents
1. [Introduction](#Introduction)
2. [Prerequisites](#Prerequisites)
3. [Data Preparation](#Data-Preparation)
4. [Model Evaluation](#Model-Evaluation )
5. [Running the Code](#Running-the-Code)
6. [References](#References)
7. [Contributors](#Contributors)

# Introduction
The introduction to the Heart Failure Prediction Using Artificial Neural Networks (ANN) project emphasizes the critical need for timely heart failure prediction, highlighting the potential of machine learning, specifically ANNs, in this domain. The project aims to achieve early detection, promote personalized medicine, advance research, and have a positive impact on public health.

The choice of ANNs is justified by their ability to discern intricate patterns and non-linear relationships within complex datasets, making them well-suited for predicting heart failure. The documentation will guide users through the project structure, usage of the ANN model, and encourage customization and contributions for the advancement of predictive healthcare analytics. Overall, the project represents a fusion of cutting-edge technology and healthcare innovation.

# Prerequisites
The code requires the following Python libraries to be installed:<br />
•	numpy<br />
•	pandas<br />
•	matplotlib<br />
•	seaborn<br />
•	scikit-learn<br />
•	keras<br />

# Data Preparation
### 1. Loading the Raw Dataset:
you need to load the dataset into your Python environment. You can use a library like Pandas for this task
### 2. Exploratory Data Analysis (EDA):
EDA is essential for gaining insights into the dataset. The **describe()** method provides statistics like mean, standard deviation, and quartiles for numerical features. Checking for missing values with **isnull().sum()** is crucial, as missing data can impact the model's performance. Visualizing feature distributions helps identify patterns and potential outliers.
### 3. Data Cleaning:
Handling missing values is a critical aspect of data cleaning. In the example, missing values are imputed with the mean, but other strategies like median or dropping rows/columns might be suitable depending on the context. The removal of outliers using z-scores is demonstrated; however, the choice of outlier detection method should align with the characteristics of your dataset.
### 4. Feature Scaling:
Feature scaling ensures that numerical features are on a similar scale, preventing certain features from dominating others during training. Standardization (z-score normalization) used in the example,that centers the data around 0 with a standard deviation of 1.
### 5. Splitting Data into Features and Target:
The dataset is split into input features (X) and the target variable (y). Adjust the column names based on your specific dataset, ensuring that target_variable corresponds to the column containing the outcome variable you want to predict.
### 6. Train-Test Split:
Splitting the data into training and testing sets allows you to assess the model's performance on unseen data. The train_test_split function is used for this purpose. The test_size parameter determines the proportion of the dataset allocated to the test set, and random_state ensures reproducibility.

# Model Evaluation 
### 1. Loading the Trained Model:
After training your ANN model, you save it for future use. Loading the trained model is the first step in the evaluation process. The load_model function from TensorFlow or Keras is used for this purpose. Ensure that the path specified in model_path points to the correct location of your trained model.
### 2. Evaluating on Test Data:
The primary goal is to assess how well the model generalizes to new, unseen data. The predict method is used to obtain model predictions on the test set (X_test). Binary predictions are typically derived by thresholding the output probabilities (0.5 in this case). Common evaluation metrics include:
### Accuracy: 
The proportion of correctly classified instances.
### Confusion Matrix: 
A table that summarizes the true positive, true negative, false positive, and false negative predictions.
### 3. ROC Curve and AUC:
Explanation: The Receiver Operating Characteristic (ROC) curve is a graphical representation of the model's ability to distinguish between classes at various threshold settings. The Area Under the Curve (AUC) score quantifies the model's discriminative power. A higher AUC indicates better performance. The visualization aids in choosing an appropriate threshold based on the desired balance between true positive and false positive rates.

# Running the Code
To run the code, ensure that the required Python libraries are installed. Then, execute the provided code in a Python environment that supports the required libraries.
# References
Article <br/>
https://www.researchgate.net/publication/364949647_Enhanced_accuracy_for_heart_disease_prediction_using_artificial_neural_network <br/>
https://ieeexplore.ieee.org/document/9112443<br/>
Dataset<br/>
https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
# Contributors
•	Hari krishna Para[https://github.com/HariKrishnaUNH] and  Veda Samohitha Chaganti[https://github.com/samohitha]<br />
Feel free to customize and enhance the code as needed for your specific use case.
