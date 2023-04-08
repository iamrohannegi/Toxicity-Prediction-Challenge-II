# Toxicity-Prediction-Challenge-II



#Code

Solution coded on the following Kaggle's notebook.
Kaggle: https://www.kaggle.com/code/x2022fic/x2022fic-submission
Libraries used: Pandas, sklearn, rdkit, xgboost


#Introduction

The objective of this project is to develop a machine learning model to predict the toxicity of chemicals. We will use XGBoost classifier with RDKit descriptors to train the model. RDKit is a collection of cheminformatics and machine learning tools for drug discovery, and XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.

#Data
We have two datasets: train.csv and test.csv. Both datasets contain information about chemicals and assay id. The train dataset has 75383 unique values, and the test dataset has 10994 unique values. We will use the train dataset to train our model and then use this model to do predictions on the test dataset.

#Preprocessing
We first load the train and test datasets into Pandas dataframes using the read_csv function. We then extract the assay_id and chemical_id from the Id column of the dataframes. 

We remove rows with Si from the train dataset, as this chemical is known to cause errors in the RDKit descriptors. Finally, we select a list of 2D RDKit descriptors to use as features in our model.

After selecting the RDKit Descriptors we need as features, we go through each row in the train dataset and generate values for the descriptors based on the chemical id we extracted before.

The predicted column in the train dataset contains values in 1s and 2s. Using the LabelEncoder, we encode the 1s and 2s to 0s and 1s, so that we can use the data in our XGBClassifier model. 

#Features Selection
After generating values for the 205 RDKit 2D Descriptors that we selected earlier for each row, we combine the names of all the 205 descriptors with assay_id that we extracted from the train dataset in the beginning. 

#Model Selection
For this submission, XGBClassifier was used. The classifier uses an ensemble of decision trees to make predictions, with each subsequent tree trained to correct the errors of the previous ones. XGBClassifier has several hyperparameters that can be tuned to optimize model performance, such as the learning rate, n_estimators, and random_state. It also supports early stopping, which can save computation time and prevent overfitting.

After a few iterations of testing, we tuned the parameters of our XGBClassifier model to include n_estimators=3000, learning_rate=0.09, early_stopping_rounds=7, and random_state=0, based on the results of the F1 score. 

#Testing the model
We tested the XGBClassifier model using the train_test_split function from the sklearn.model_selection module. This function randomly splits the dataset into two subsets: the training set and the testing set. 

The training set is used to fit the XGBoost model, while the testing set is used to evaluate the performance of the model. We then calculate the f1-score of the model, which you can import the f1_score function from the sklearn.metrics module. 

Once the model is trained, we use it to make predictions on the testing set. Then,  using the f1_score function we calculate the f1-score of the predicted labels and the actual labels from the testing set.


Following are the results of the internal evaluation F1 score and the public score of each submission.

![image](https://user-images.githubusercontent.com/16450711/230736721-3b190fb2-812b-4462-87b3-4eadac6335bd.png)

#Predictions
Finally, after tuning the parameters of our XGBClassifier model using the train_test_split evaluation, we fitted the model with the entire data from the train dataset for predictions. 

At the end, we inverted the prediction back to 1s and 2s using our label encoder, as the submission file uses values 1s and 2s instead of 0s and 1s.

