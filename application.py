from flask import Flask, request, render_template,jsonify
import joblib 
import pandas as pd
model=joblib.load('final_model.pkl')
vect=joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('web.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get the form data
    # Process the data (for this example, just return it)
    data=model.predict(vect.transform(pd.Series((request.form.get(('user-input'))))))
    # Render the result in the respons
    return str(data)

if __name__ == '__main__':
    app.run(debug=True)
