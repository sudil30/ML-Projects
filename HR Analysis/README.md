# HR Analytics- Job change of data scientists: Project Overview 
* Created a RandomForest model(Acc.=85.1%) that predicts whether a data scientist is looking for a job change or not to help HR in categorizing candidates and reducing costs during hiring.
* Dealt with ordinal categorical features by manually encoding, and one hot encoding other categorical features.
* Applied KNNImputer to handle missing data.
* Used SMOTE(Synthetic Minority Oversampling Technique) to balance the Imbalanced dataset.
* Optimized Random Forest Classifier using RandomizedSearchCV and GridsearchCV to reach the best model. 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, imblearn, pandas_profiling    
**GridSearch:** https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74   
**SMOTE** https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/  

## Dataset
The dataset is taken from Kaggle   
Link: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists   
**Description**   
With the help of these features we have to predict which data scientist is looking for a job change and thus, is a better prospect in terms of hiring.  

* enrollee_id : Unique ID (Useless here)
* city: City code
* city_ development _index : Developement index of the city (scaled)
* gender: Gender of candidate
* relevent_experience: Relevant experience of candidate
* enrolled_university: Type of University course enrolled if any
* education_level: Education level of candidate
* major_discipline :Education major discipline of candidate
* experience: Total experience in years
* company_size: No of employees in current company
* company_type : Type of current company
* lastnewjob: Difference in years between previous job and current job
* training_hours: training hours completed

target: 0 – Not looking for job change, 1 – Looking for a job change

## Data Preprocessing

* Label encoded 'city' column
* Manually encoded Ordinal categorical features
* One hot encoded other categorical features
* Used KNN Imputer for missing data
* Used SMOTE to balance the data by oversampling

## EDA
For EDA, I used a library "pandas_profiling" which gave an all round report of the dataset.
Here are some images from the same.
![alt text](https://github.com/sudil30/ML-Projects/blob/main/HR%20Analysis/Resources/Correlation.jpg "Correlations")
![alt text](https://github.com/sudil30/ML-Projects/blob/main/HR%20Analysis/Resources/Missing%20data.jpg "Missing Data")
![alt text](https://github.com/sudil30/ML-Projects/blob/main/HR%20Analysis/Resources/Imbalanced%20target%20feature.jpg "Imbalanced Target feature")

## Model Selection

First, I split the data into train and tests sets with a test size of 20%.   

I tried three different models:
*	**Decision Tree Classifier** : 79.6%
*	**SVM**: 80.1%
*	**Random Forest**: 84.2%

So, as the random forest classifier performed the best out of these I furthur tuned the hyperparameters.

## Hyperparameter Tuning
First, I used RandomizedSearchCV to give a better set of parameters and with the help of these, created a parameter grid. Then, I used GridSearchCV to maximize the performance of the model.   

**Model Accuracy: 85.1%**

