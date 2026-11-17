Social Network Ads: Purchase Prediction (Flask + Machine Learning)

A Flask web application that predicts whether a user will purchase a product after seeing a social media advertisement, based on their gender, age and estimated salary. The prediction is made by a Random Forest classifier trained on the Social Network Ads dataset.

Demo

Add a screenshot of the form and the result page here, for example static/screenshot.png:

![App screenshot](static/screenshot.png)

Features
Web form that takes gender, age and estimated salary
Instant prediction: will purchase or will not purchase
Input validation on the form (number fields with limits) and on the server (invalid input returns a friendly message instead of an error)
Scaling and model are saved together in one Scikit-learn Pipeline, so the app always preprocesses input exactly as in training
Tech Stack
Area	Tools
Language	Python 3
Web	Flask, HTML5, CSS3, Jinja2 templates
Machine learning	Scikit-learn (RandomForestClassifier, StandardScaler, Pipeline)
Data	Pandas
Model storage	Pickle
Dataset

Social Network Ads dataset: 400 records with User ID, Gender, Age, EstimatedSalary and the target Purchased (0 = no, 1 = yes). It has no missing values.

Model
Pre-processing: Gender is encoded (Male = 1, Female = 0). User ID is dropped because it is only an identifier and carries no predictive information.
Split: 80% training and 20% testing (stratified, random_state=42).
Pipeline: StandardScaler followed by RandomForestClassifier (100 trees, entropy criterion).
Evaluation: accuracy, classification report, confusion matrix and 5-fold cross-validation.
Results
Metric	Value
Test accuracy	[X]%
5-fold CV mean accuracy	[X]%

Replace the values with the output of your notebook. Cross-validation is the more reliable figure because the test set contains only 80 rows.

Project Structure
social-media-ad-campaign-prediction/
├── app.py                         Flask application (loads the model and serves predictions)
├── Social_Media_AD_Campaign.ipynb Training and evaluation notebook
├── model.pkl                      Saved Scikit-learn pipeline (scaler + model)
├── requirements.txt               Python dependencies
├── templates/
│   ├── index.html                 Input form
│   └── result.html                Prediction result page
└── static/
    └── style.css                  Styling
Getting Started
1. Clone the repository
bash
git clone https://github.com/bintabinty/social-media-ad-campaign-prediction.git
cd social-media-ad-campaign-prediction
2. Create a virtual environment and install dependencies
bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
3. Run the app
bash
python app.py

Open http://127.0.0.1:5000 in your browser, enter the details and click Predict.

(Optional) Retrain the model

Open Social_Media_AD_Campaign.ipynb, place Social_Network_Ads.csv in the same folder and run all cells. This regenerates model.pkl.

Use the same scikit-learn version for training and running the app, otherwise the pickle file may fail to load.

Requirements

requirements.txt:

flask
pandas
scikit-learn
Limitations
The dataset is small (400 rows), so the model is a learning project, not a production system.
Only three features are used; real ad-conversion models need richer behavioural data.
The development server (python app.py) is for local use only.
Future Improvements
Compare more models (Logistic Regression, SVM, XGBoost) with hyperparameter tuning
Show prediction probability in the result page
Deploy on Render or Hugging Face Spaces and add the live link here
Author

Binty Biju LinkedIn | GitHub
