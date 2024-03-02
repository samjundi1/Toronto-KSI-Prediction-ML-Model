# Executive Summary
In this project, our objective was to predict the likelihood of fatal cases in traffic collisions in Toronto using the KSI dataset, which includes information about all traffic collision events where a person was either Killed or Seriously Injured from 2006 to 2021. To achieve this, our team has developed and compared four machine learning models - Random Forest, Histogram-based Gradient Boosting, Logistic Regression, and Support Vector Machines (SVM) with Bagging - to predict fatal traffic collision cases in Toronto. Our DataLoader class loads the dataset and performs data cleaning, transformation, standardization, SMOTE sampling, feature selection, and encoding.
We evaluated the performance of the models using accuracy, precision, recall, and F1-score metrics. The Histogram-based Gradient Boosting model demonstrated the best performance, with an accuracy of 96.46%, a precision of 96.42%, a recall of 96.46%, and an F1-score of 96.31%.
# Overview of Solution
Our solution involved the creation of a DataLoader class to efficiently manage the KSI dataset, preprocess the data, and prepare it for modelling. This class was responsible for loading the dataset, cleaning the data, transforming features, selecting relevant features, applying SMOTE sampling, standardizing the data, and encoding categorical variables.
We then built four machine learning models - Random Forest, Histogram-based Gradient Boosting, Logistic Regression, and SVM with Bagging - to predict whether a traffic collision would result in a fatality. We trained each model on the preprocessed data and evaluated it using accuracy, precision, recall, and F1-score metrics. The performance of each model was as follows:

| | Accuracy | Precision |	Recall |	F1 Score|
|-|-|-|-|-|
| Random Forest |	92.51%	| 92.07%	| 92.51%	| 91.96% |
| Histogram-based Gradient Boosting |	96.46%	| 96.42%	| 96.46%	| 96.31% |
| Logistic Regression |	61.52%	| 80.75% | 61.52%	| 67.41% |
| SVM With Bagging | 61.92% |	80.52% | 61.92% | 67.73% |

![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/ceabf30f-9faa-40e5-9ebf-608ab9e8790b)
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/15e3c9bc-2204-4ef2-b9b3-ebd5cda1d467)


Based on these results, the Histogram-based Gradient Boosting model outperformed the other models, making it the most suitable choice for predicting fatal traffic collisions in Toronto using the KSI dataset.

# Data Exploration and Findings
## Data Exploration
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/867920c0-92a7-4283-9ed5-a22ae7f74ae3)
57 columns exist with 17488 rows and not much to find on this part of exploration.

![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/e3a342cd-f404-4249-ae2a-bf4a8be8593e)

The Dataset is a mixture of numerical and categorical fields, and we can see the rows that have a lot of null values.

![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/efa66a6b-35c4-4cd2-8a96-e43bdc1ccb2e)

Here we can see which fields have a lot of unique values so we can remove them from the feature selection.

## Findings
We found out that 90% of the collision is non-fatal and around 10% is fatal. We also discovered that there are some fields that can be a red herring for the model prediction such as Injury fields which contains major and minor classes. There are also a lot of yes or no features that can be fixed by populating the null value with no. In addition, there is another category which is property damage that can be non-fatal.

# Data Preprocessing and Feature Selection
In this project, we utilized the KSI dataset containing traffic collision events that resulted in either fatalities or severe injuries in Toronto from 2006 to 2021. The DataLoader class was responsible for loading the dataset, cleaning it, transforming it, selecting essential features, and preparing the data for modelling. The data cleaning involved handling missing values, extracting new features, and dropping unnecessary columns. The new features, such as "MONTH" and "DAY_WEEK," were derived from the "DATE" column.
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/a1e748da-ee95-4597-8003-325ade917a7f)
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/47049d8b-2114-47fe-9fd3-51bbc0c125d9)


We created a column transformation pipeline to streamline feature preprocessing, which included imputing missing values, scaling numerical features, and encoding categorical features. 
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/3923d5ba-8352-479d-9f74-bd0412111038)
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/ecafb54e-d37c-47dc-8d90-50af3aab17e4)
![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/d542c5fd-0b51-4a07-8646-d98505f8b511)

After preprocessing the dataset, we used a Random Forest classifier to determine the most important features, which were then used for further analysis.

![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/3f3ec9e2-6f66-412a-a3b8-2a1463674f05)

It shows that the road condition is the most important feature, followed by disability, division, month, red light, aggressive driver, and light.

![image](https://github.com/DerDangla/Toronto-KSI-Prediction-ML-Model/assets/8519156/5589ce80-eb0d-40a1-a96e-4f15ab7dc38e)

We have filtered this by selecting only the features with importance greater than 0.017. As you can see in the screenshot above. Only 18 features remain.

# Model Building
The Model Building phase involved training and evaluating various classifiers on the preprocessed and feature-selected dataset. The DataLoader class was responsible for splitting the dataset into training and testing sets, ensuring a balanced class distribution by applying the Synthetic Minority Over-sampling Technique (SMOTE) to the training set, and saving the data for subsequent model training. We built four models to predict fatal cases: Random Forest, Histogram-based Gradient Boosting, Logistic Regression, and Support Vector Classifier (SVC) with bagging. We evaluated these models based on their performance metrics, such as the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC). By comparing the performance of the different models, we aimed to identify the most effective model for predicting traffic collision outcomes in the city of Toronto. 

## Data Sources
-	Train Data: We used a dataset containing historical customer information from the telecom company, including demographic, billing, and usage data. The dataset consists of 15,000 records with a churn rate of 20%.
-	Test Data: A separate dataset of 5,000 records was used for evaluating model performance. This dataset exhibits a similar churn rate as the training data.

## Model Building Assumptions
-	The historical data is representative of future customer behavior.
-	The train and test datasets are assumed to come from the same population.
-	The features included in the dataset have a direct or indirect impact on customer churn.
-	The churn rate in the historical data is representative of the true churn rate.

## Data Preprocessing
-	Missing values were imputed with median values for continuous variables and mode values for categorical variables.
-	Categorical variables were one-hot encoded to create dummy variables.
-	Feature scaling was applied to normalize continuous variables.
-	The train dataset was split into a training set (80%) and a validation set (20%) using stratified sampling to maintain the same churn rate across all subsets. 

## Algorithms Tested
-	Logistic Regression
-	Decision Trees
-	Random Forest
-	Support Vector Machines
-	Neural Networks 

## Model Evaluation Metrics
-	Accuracy
-	Precision
-	Recall
-	F1-score
-	Area Under the Receiver Operating Characteristic (ROC) Curve (AUC-ROC)






