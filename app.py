from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pyrebase
import io
import os
import predictFinal

config = {
    "apiKey": "AIzaSyCNREzzNW83QsUIgG6QzX-ozpOINWL955c",
    "authDomain": "security-whale.firebaseapp.com",
    "databaseURL": "https://security-whale.firebaseio.com",
    "projectId": "security-whale",
    "storageBucket": "security-whale.appspot.com",
    "messagingSenderId": "774091153639",
    "appId": "1:774091153639:web:a54a1397c23f22242e0242",
    "measurementId": "G-6VD698LMTY"
}

firebase = pyrebase.initialize_app(config)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        data = request.data
        data_as_string = str(data, "utf-8")

        data_as_string = list(data_as_string.split('x23model!t@ype!x56'))
        if not data:
            return jsonify({"error":"no file"})
        return predictFinal.makePrediction(data_as_string[0], data_as_string[1]), 201
    else:
        return jsonify({"about":"Nothing to see here..."})

# def index():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
