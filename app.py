from flask import Flask,render_template,url_for,request
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

app= Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	iris=datasets.load_iris()
	iris.data[0:5]
	iris.target
	data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
	})
	X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
	y=data['species']  # Labels
	# Split dataset into training set and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
	#Create a Gaussian Classifier
	clf=RandomForestClassifier(n_estimators=100)
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	if request.method == 'POST':
	 	sepal_length=request.form['sepal_length']
	 	sepal_width=request.form['sepal_width']
	 	petal_length=request.form['petal_length']
	 	petal_width=request.form['petal_width']
	 	data1=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],columns=['sepal length', 'sepal width', 'petal length', 'petal width'],dtype=float)
	 	my_prediction=clf.predict(data1)

	return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)	