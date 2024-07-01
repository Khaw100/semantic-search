from flask import Flask, render_template, request, redirect, url_for, jsonify
from loadModel import load_model
from ModelClasses import TFIDFModel, Model1, Model2, Model3, Model4
import json

app = Flask(__name__)

# Load the models
model_dict = {
    "model1": load_model("Model/tfidf_model.pxl"),
    "model2": load_model("Model/tfidf_model.pxl"),
    "model3": load_model("Model/tfidf_model.pxl"),
    "model4": load_model("Model/tfidf_model.pxl"),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/searchresults', methods=['POST'])
def search_results():
    search_text = request.form['searchText']
    model_choice = request.form['model']
    model_key = model_choice.lower().replace(" ", "")
    model = model_dict.get(model_key)
    
    if model:
        results = model.search(search_text)  # Assuming your model has a search method
        return render_template('results.html', searchText=search_text, model=model_choice, results=results, page=1, total_pages=1)
    else:
        return render_template('results.html', searchText=search_text, model=model_choice, results=[], page=1, total_pages=1)

@app.route('/submit', methods=['POST'])
def submit_feedback():
    feedback_data = {
        'searchText': request.form['searchText'],
        'model': request.form['model'],
        'relevant': request.form.getlist('relevant'),
        'not_relevant': request.form.getlist('not_relevant')
    }

    with open("feedback.json", "w") as f:
        json.dump(feedback_data, f, indent=4)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
