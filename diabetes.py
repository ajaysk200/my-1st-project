import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


url = 'https://www.dropbox.com/scl/fi/0uiujtei423te1q4kvrny/diabetes.csv?rlkey=20xvytca6xbio4vsowi2hdj8e&raw=1'

# load the dataset
diabetes_dataset = pd.read_csv(url)

# preview data
print(diabetes_dataset.head())
diabetes_dataset.shape
# getting the statistical measurement of data 
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())
# seprating data and leberl
x = diabetes_dataset.drop(columns = 'Outcome', axis =1)
y = diabetes_dataset['Outcome']
print(x)
print(y)
# Data standardization
scaler = StandardScaler()
scaler.fit(x)
scaler.fit_transform(x)
standardized_data = scaler.fit_transform(x)
print(standardized_data)   
x = standardized_data     #represents data
y = diabetes_dataset['Outcome']  # represents the model
print(x)
print(y)
# now we split the data into training data and test data 
x_train , x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y, random_state = 2)
print(x.shape, x_train.shape, x_test.shape)
# training the model 
classifier  = svm.SVC(kernel = 'linear')
# training the support vector machine classifier 
classifier.fit(x_train, y_train)
# Model evaluation
# accuracy accuracy_score 
# accuracy_score on the training data 
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('accuracy score of the training data : ', training_data_accuracy)


# accuracy_score on the test data 
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('accuracy score of the test data : ', test_data_accuracy)
# Making a predictive System 



input_data = (5,166,72,19,175,25.8,0.587,51)
# changing the input data to numpy array 
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# sandardize the input data 
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction  = classifier.predict(std_data)
print(prediction) 

if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
