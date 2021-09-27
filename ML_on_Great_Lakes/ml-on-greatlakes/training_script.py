from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load dataset- CHANGE THIS TO LOCAL FILENAME
filename = '/scratch/hpcstaff_root/hpcstaff/richeym/workshops/ml-on-greatlakes/iris.csv'

#Specify column names of data
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#Read the data from scratch into our workspace
dataset = read_csv(filename, names=names)

#Divide the data between training and validation
array = dataset.values
features = array[:, 0:4]
classification = array[:,4]
features_train, features_validate, classification_train, classification_validate = train_test_split(features, classification, test_size=0.2, random_state=1, shuffle=True)

#Create a list of models that we want to compare performance on
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#Evaluate each of the models and store the results in a list
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, features_train, classification_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	mean = cv_results.mean()
	std = cv_results.std()
	print(f'{name}: {mean} ({std})')
