# 🎵 Song Popularity Predictor

Welcome to the **Song Popularity Predictor**, a Flask-based app trained on a Spotify dataset to forecast how well a track will perform and offer tips to boost its popularity.

## 🚀 Key Features

- **Popularity Prediction**: Enter an existing song or add a new one (with features like danceability, energy, key, loudness, etc.) to see its predicted popularity.
- **Improvement Suggestions**: Get actionable insights on which audio features to tweak for a more hit-worthy track.
- **Artist Explorer**: Quickly find and browse songs by your favorite artists.
- **Data Analysis**: Dive into the dataset to discover the top 10 artists, genres, and more.
- **Model Visualization**: View performance charts to understand how well our prediction model works.

## 🗄️ Dataset

Our model is trained on a released Spotify CSV dataset containing song attributes such as:

- **danceability**
- **energy**
- **key**
- **loudness**
- _…and many more!_

## ⚙️ Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/song-popularity-predictor.git
   cd song-popularity-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the App

```bash
python app.py
``` 

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

## 📄 Project Structure

- **app.py**: Main Flask server
- **templates/**: HTML pages for Home, Analysis, and Visualization
- **static/**: CSS, JavaScript, and image assets
- **data/spotify.csv**: The raw dataset
- **models/**: Saved prediction model files

---

Built with ❤️ and Spotify data. Feel free to contribute or send feedback!
