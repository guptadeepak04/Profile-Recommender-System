from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the models safely
try:
    with open(r'D:\aluminai network project\profile_recommender\scaler_model.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(r'D:\aluminai network project\profile_recommender\knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)  # Stop execution if models fail to load

# Load your alumni data safely
try:
    data = pd.read_csv(r'D:\aluminai network project\profile_recommender\your_alumni_data.csv')
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit(1)

features = ['Graduation Year', 'Years of Experience']


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        university_name = request.form['university_name']
        recommendations = recommend_alumni_by_university(university_name)

    return render_template('index.html', recommendations=recommendations)


def recommend_alumni_by_university(university_name):
    university_data = data[data['University'] == university_name]

    if university_data.empty:
        return [{"message": f"No alumni found from {university_name}"}]  # Fixed issue

    university_features = university_data[features]

    try:
        university_features_scaled = scaler.transform(university_features)
        distances, indices = knn.kneighbors(university_features_scaled, n_neighbors=5)
    except Exception as e:
        return [{"error": f"Model error: {e}"}]

    recommended_alumni = data.iloc[indices.flatten()]

    return recommended_alumni[
        ['Name of Student', 'University', 'Job Title', 'Company', 'Technical and Non-technical Skills']].to_dict(
        orient='records')


if __name__ == '__main__':
    app.run(debug=True)
