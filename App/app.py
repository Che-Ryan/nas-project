from flask import Flask,flash,render_template,url_for,request,redirect
import summarizer as s
import fileload as fl

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        article = request.form['inputdata']
        if len(article)==0:
            f = request.files['file']
            article = fl.fileload(f)            
        r = 0.5
        triplets,sentences,processed = s.parsing(article)
        summary = s.summarize(triplets,sentences,processed,r)
        cat,_ = s.predict_cat(article)
    return render_template('result.html',summary = summary,category=cat)

if __name__ == '__main__':
    app.run(debug=True)
