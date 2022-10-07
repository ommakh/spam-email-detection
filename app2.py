import pickle
import numpy
import flask
from flask import Flask,render_template,request
clf = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))
app = Flask(__name__)



@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
  if request.method == 'POST':
      message = request.form.get(('message'))
      data = [message]
      vect = cv.transform(data).toarray()
      my_prediction = clf.predict(vect)
  return render_template('result.html',prediction = my_prediction)
if __name__=='__main__':
    app.run(debug=True)