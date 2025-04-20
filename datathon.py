import matplotlib
matplotlib.use('Agg')  # Use a headless backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import io
import base64

# -------------------- Step 1: Load and Prepare Dataset --------------------
try:
    # Update the CSV file path if needed.
    df = pd.read_csv("/users/chiragjogi/PES/SEM-6/Epoch_2.0/spotify_songs/spotify_songs.csv")
    print("Columns in dataset:", df.columns.tolist())
except Exception as e:
    print("Error reading CSV:", e)
    df = None

if df is not None:
    try:
        # Create the 'artist_popularity' column as the average track popularity per artist.
        df['artist_popularity'] = df.groupby('track_artist')['track_popularity'].transform('mean')
    except Exception as e:
        print("Error processing artist_popularity:", e)
    
    # Define features and target.
    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'artist_popularity'
    ]
    target = 'track_popularity'
    
    try:
        # Select required columns including track_id and track_name; drop rows with missing values.
        df_model = df[['track_id', 'track_name'] + features + [target]].dropna()
    except Exception as e:
        print("Error processing dataframe:", e)
        df_model = None
else:
    df_model = None

# -------------------- Step 2: Train a Regression Model --------------------
if df_model is not None:
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
else:
    mse, r2, X_test, y_test, reg = None, None, None, None, None

# -------------------- Step 3: Get Song Details, Prediction, and Recommendations --------------------
def get_song_details(song_name):
    if df_model is None or reg is None:
        return {"error": "Model not properly initialized."}
    try:
        # Search for matching song (case-insensitive)
        song_match = df_model[df_model['track_name'].str.contains(song_name, case=False, na=False)]
    except Exception as e:
        return {"error": f"Error during song search: {str(e)}"}
    if song_match.empty:
        # Signal that this is a new song.
        return {"new_song": True, "song_name": song_name}
    try:
        song_row = song_match.iloc[0]
        song_feature_values = song_row[features].to_frame().T
        predicted_popularity = reg.predict(song_feature_values)[0]
        actual_popularity = song_row[target]
        absolute_error = abs(predicted_popularity - actual_popularity)
        percent_error = (absolute_error / actual_popularity * 100) if actual_popularity != 0 else float('inf')
        popularity_status = "Popular" if predicted_popularity >= 60 else "Not Popular"
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}
    try:
        popularity_median = df_model[target].median()
        popular_songs = df_model[df_model[target] >= popularity_median]
        avg_values_popular = popular_songs[features].mean()
        recommendations = {}
        for feat in features:
            if feat == 'artist_popularity':
                continue
            current_val = song_row[feat]
            avg_val = avg_values_popular[feat]
            if current_val < avg_val:
                recommendations[feat] = f"Consider increasing {feat} (current: {current_val:.2f}, average: {avg_val:.2f})."
            elif current_val > avg_val:
                recommendations[feat] = f"Consider decreasing {feat} (current: {current_val:.2f}, average: {avg_val:.2f})."
            else:
                recommendations[feat] = f"{feat} is already optimal (value: {current_val:.2f})."
    except Exception as e:
        return {"error": f"Error during recommendations: {str(e)}"}
    try:
        feature_vector = song_row[features].tolist()
        similar_songs = get_similar_songs(feature_vector, top_n=5)
    except Exception as e:
        similar_songs = []
    result = {
        "track_name": song_row["track_name"],
        "predicted_popularity": predicted_popularity,
        "actual_popularity": actual_popularity,
        "popularity_status": popularity_status,
        "absolute_error": absolute_error,
        "percent_error": percent_error,
        "recommendations": recommendations,
        "similar_songs": similar_songs,
        "mse": mse,
        "r2": r2
    }
    return result

# -------------------- New Function: Get Similar Songs --------------------
def get_similar_songs(new_feature_vector, top_n=5):
    if df_model is None:
        return []
    try:
        X_features = df_model[features].values
        new_vector = np.array(new_feature_vector)
        distances = np.linalg.norm(X_features - new_vector, axis=1)
        top_indices = np.argsort(distances)[:top_n]
        similar_songs = df_model.iloc[top_indices][['track_name', 'track_artist', 'track_popularity']].to_dict(orient='records')
        return similar_songs
    except Exception as e:
        print("Error generating similar songs:", e)
        return []

# -------------------- New Function: Get Songs by Artist --------------------
def get_songs_by_artist(artist_name):
    try:
        songs = df[df['track_artist'].str.contains(artist_name, case=False, na=False)]
        if songs.empty:
            return None
        return songs[['track_name', 'track_popularity']].to_dict(orient='records')
    except Exception as e:
        print("Error in get_songs_by_artist:", e)
        return None

# -------------------- Other Visualization Functions --------------------
def get_top_artists_chart():
    try:
        top_artists = df.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_artists.values, y=top_artists.index, palette="viridis")
        plt.xlabel("Average Popularity")
        plt.ylabel("Artist")
        plt.title("Top 10 Artists by Average Track Popularity")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print("Error generating top artists chart:", e)
        return None

def get_top_genres_chart():
    try:
        top_genres = df.groupby('playlist_genre')['track_popularity'].mean().sort_values(ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_genres.values, y=top_genres.index, palette="magma")
        plt.xlabel("Average Popularity")
        plt.ylabel("Genre")
        plt.title("Top 10 Genres by Average Track Popularity")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print("Error generating top genres chart:", e)
        return None

def get_top_songs_list():
    try:
        top_songs = df.sort_values('track_popularity', ascending=False).drop_duplicates(subset='track_name').head(10)
        songs_list = top_songs[['track_name', 'track_artist', 'track_popularity']].to_dict(orient='records')
        return songs_list
    except Exception as e:
        print("Error generating top songs list:", e)
        return None

def get_top_songs_by_artist_list():
    try:
        top_artists = df.groupby('track_artist')['track_popularity'].mean().sort_values(ascending=False).head(10).index.tolist()
        result = {}
        for artist in top_artists:
            df_artist = df[df['track_artist'] == artist].sort_values('track_popularity', ascending=False)\
                        .drop_duplicates(subset='track_name').head(5)
            songs = df_artist[['track_name', 'track_popularity']].to_dict(orient='records')
            result[artist] = songs
        return result
    except Exception as e:
        print("Error generating top songs by artist list:", e)
        return None

def get_bar_graph():
    try:
        n_samples = 20
        indices = np.arange(n_samples)
        actual = y_test.values[:n_samples]
        predicted = y_pred[:n_samples]
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(indices - width/2, actual, width, label='Actual Popularity', color='blue')
        plt.bar(indices + width/2, predicted, width, label='Predicted Popularity', color='red')
        plt.xlabel("Test Sample Index")
        plt.ylabel("Popularity")
        plt.title("Bar Graph: Predicted vs. Actual Track Popularity")
        plt.xticks(indices, indices)
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print("Error generating bar graph:", e)
        return None

def get_project_scatter_plot():
    try:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Actual Popularity")
        plt.ylabel("Predicted Popularity")
        plt.title("Scatter Plot: Actual vs. Predicted Popularity")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print("Error generating project scatter plot:", e)
        return None

def get_custom_accuracy():
    try:
        tolerance = 15
        accurate_predictions = abs(y_pred - y_test) < tolerance
        custom_accuracy = accurate_predictions.mean()
        return custom_accuracy
    except Exception as e:
        print("Error computing custom accuracy:", e)
        return None
