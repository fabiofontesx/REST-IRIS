

# REST-IRIS

### Resum
REST-IRIS is an Flask-Restx based API with an Artificial Neural Network that's receive the parameters of [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) and return the classified iris description

### Project structure
```
project/
├── app.py
├── iris.csv
├── classificador_iris.json
├── iris_weigths.h5
├── requiriments.txt
└── controllers
    ├── iris_classifier_controller.py
└── models
	├── iris_model.py
	├── model.py
└── service
	├── iris_classifier_service.py.py
```

### Instructions
On root path

 1. **Install dependencies**
```
	pip install -r requiriments.txt
```
 2. **Execute application**
```
	python app.py
```
On the first time, the application will try to load the structure and the weights of neural network, if can't find the respective files, will be realize the training and save the structure on **classificador_iris.json** and the weigths on **iris_weigths.h5**.
After all initalizations the application will be expose on Flask default address http://127.0.0.1:5000 and will make available the Swagger Documentation.

