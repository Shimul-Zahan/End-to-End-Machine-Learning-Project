from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# start out appliaction
application = Flask(__name__)   # this is our entry point
app = application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        pass
        
