from flask import Flask, request, render_template
import numpy as np
import pickle
from gtts import gTTS
import os
import time

# Language for audio
language = 'en'

# Importing model and scalers
model = pickle.load(open(r'Crop-Recommendation-System-Using-Machine-Learning\model.pkl', 'rb'))
sc = pickle.load(open(r'Crop-Recommendation-System-Using-Machine-Learning\standscaler.pkl', 'rb'))
ms = pickle.load(open(r'Crop-Recommendation-System-Using-Machine-Learning\minmaxscaler.pkl', 'rb'))

# Creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Fetch user inputs
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Prepare input features
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale input features and make prediction
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    # Define crop dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
        11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
        15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Generate result and audio output
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{}".format(crop)
        mytext = "The best crop to be cultivated is {}".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        mytext = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Generate and save the audio file (static filename)
    audio_path = "static/op.mp3"
    myobj = gTTS(text=mytext, lang=language, slow=False)
    if not os.path.exists('static'):
        os.makedirs('static')
    myobj.save(audio_path)

    # Append timestamp to audio file URL to prevent caching
    audio_file_url = f"{audio_path}?t={int(time.time())}"

    # Render the result and audio
    return render_template('index.html', result=result, audio_file=audio_file_url)

# Run the flask app
if __name__ == "__main__":
    app.run(debug=True)
