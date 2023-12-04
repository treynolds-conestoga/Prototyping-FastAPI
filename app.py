# Packages
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import List
import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, jsonify, request

app = Flask(__name__)

def create_model()->None:
    # Load Data
    iris = load_iris()
    
    features = pd.DataFrame(iris.data,columns=iris.feature_names)
    target = pd.DataFrame(iris.target,columns=['y'])
    
    # Split data into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        target, 
        test_size=0.4, 
        random_state=1
    )
    
    # Train Model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # save model
    with open("./model/model.pickle", "wb") as file:
        pickle.dump(knn, file)
    
    return

def predict(inputs:np.ndarray)->List[int]:     
    # Train
    if not os.path.isfile("./model/model.pickle"):
        print("Model Created")
        create_model()
            
    # get model
    model = None
    with open('./model/model.pickle', 'rb') as file:
        model = pickle.load(file)
       
    # Return list or predictions 
    return model.predict(inputs).tolist()

@app.route("/", methods = ['GET'])
def main_page():
    return jsonify({"message": "App PAge Viewable"})

@app.route("/predict", methods = ['POST'])
def do_prediction():
    api_inputs = pd.DataFrame(request.get_json(), index=[0])
    print(api_inputs)
    
    # Make Predictions
    pred = predict(api_inputs)
    print(pred)
    
    return jsonify({
        "inputs": api_inputs.to_json(orient = 'records'),
        "pred": pred
    })


if __name__ == "__main__":
    print("\nStarting Program:")
    print("=================")
    
    app.run(host='0.0.0.0', port=5051)
    
    # Load Data
    #iris = load_iris()
    
    #features = pd.DataFrame(iris.data,columns=iris.feature_names)
    #target = pd.DataFrame(iris.target,columns=['y'])
    
    # Split data into test and train
    #X_train, X_test, y_train, y_test = train_test_split(
    #    features, 
    #    target, 
    #    test_size=0.4, 
    #    random_state=1
    #)
    
    # Predict test data
    #y_pred = predict(X_test)
    #print(y_pred)
    

