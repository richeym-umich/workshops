from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Load our dataset- CHANGE TO LOCAL FILENAME
filename = '/scratch/hpcstaff_root/hpcstaff/richeym/workshops/ml-on-greatlakes/iris.csv'

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(filename, names=names)

#Split out validation set
array = dataset.values
features = array[:, 0:4]
classification = array[:,4]
features_train, features_validation, classification_train, classification_validation = train_test_split(features, classification, test_size=0.2, random_state=1)

#Make predictions on validation dataset with SVC
model = SVC(gamma='auto')
model.fit(features_train, classification_train)
predictions = model.predict(features_validation)

#Evaluate our predictions
print('\n-----ACCURACY SCORE-----')
print(accuracy_score(classification_validation, predictions))
print('\n-----CONFUSION MATRIX-----')
print(confusion_matrix(classification_validation, predictions))
print('\n-----CLASSIFICATION REPORT-----')
print(classification_report(classification_validation, predictions))
