from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import io
import os
import myTest

firebase = pyrebase.initialize_app(config)


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        data = request.data
        data_as_string = str(data, "utf-8")
        if not data:
            return jsonify({"error":"no file"})
        return myTest.makePrediction(data_as_string), 201
    else:
        return jsonify({"about":"Hello World!"})

# def index():
#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))