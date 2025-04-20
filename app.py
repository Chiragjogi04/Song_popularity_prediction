from flask import Flask, render_template, request, redirect, url_for
from datathon import (
    get_song_details,
    get_top_artists_chart,
    get_top_genres_chart,
    get_top_songs_list,
    get_top_songs_by_artist_list,
    get_songs_by_artist,
    get_bar_graph,
    get_project_scatter_plot,
    get_custom_accuracy,
    mse,
    r2,
    df, df_model, reg,
    features, target
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        search_type = request.form.get("search_type")
        if search_type == "song":
            if "new_song_submission" in request.form:
                song_name = request.form.get("song_name")
                track_artist = request.form.get("track_artist")
                danceability = float(request.form.get("danceability"))
                energy = float(request.form.get("energy"))
                key = int(request.form.get("key"))
                loudness = float(request.form.get("loudness"))
                mode = int(request.form.get("mode"))
                speechiness = float(request.form.get("speechiness"))
                acousticness = float(request.form.get("acousticness"))
                instrumentalness = float(request.form.get("instrumentalness"))
                liveness = float(request.form.get("liveness"))
                valence = float(request.form.get("valence"))
                tempo = float(request.form.get("tempo"))
                duration_ms = float(request.form.get("duration_ms"))
                if track_artist in df['track_artist'].unique():
                    artist_popularity = df[df['track_artist'] == track_artist]['track_popularity'].mean()
                else:
                    artist_popularity = df['track_popularity'].median()
                feature_vector = [
                    danceability, energy, key, loudness, mode, speechiness,
                    acousticness, instrumentalness, liveness, valence, tempo, duration_ms, artist_popularity
                ]
                pred = app.config["reg"].predict([feature_vector])[0]
                popular = "Popular" if pred >= 60 else "Not Popular"
                popularity_median = df_model[target].median()
                popular_songs = df_model[df_model[target] >= popularity_median]
                avg_values_popular = popular_songs[features].mean()
                recommendations = {}
                feature_names = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
                input_features = {
                    'danceability': danceability,
                    'energy': energy,
                    'key': key,
                    'loudness': loudness,
                    'mode': mode,
                    'speechiness': speechiness,
                    'acousticness': acousticness,
                    'instrumentalness': instrumentalness,
                    'liveness': liveness,
                    'valence': valence,
                    'tempo': tempo,
                    'duration_ms': duration_ms
                }
                for feat in feature_names:
                    curr_val = input_features[feat]
                    avg_val = avg_values_popular[feat]
                    if curr_val < avg_val:
                        recommendations[feat] = f"Consider increasing {feat} (current: {curr_val:.2f}, average: {avg_val:.2f})."
                    elif curr_val > avg_val:
                        recommendations[feat] = f"Consider decreasing {feat} (current: {curr_val:.2f}, average: {avg_val:.2f})."
                    else:
                        recommendations[feat] = f"{feat} is already optimal (value: {curr_val:.2f})."
                similar_songs = []  # For new song, similar songs might not be computed
                result = {
                    "track_name": song_name,
                    "predicted_popularity": pred,
                    "popularity_status": popular,
                    "artist_popularity": artist_popularity,
                    "recommendations": recommendations,
                    "similar_songs": similar_songs
                }
                return render_template("index.html", result=result, new_song_submission=True)
            else:
                song_name = request.form.get("song_name")
                result = get_song_details(song_name)
                if result.get("new_song", False):
                    return render_template("index.html", result=result, new_song=True)
                return render_template("index.html", result=result)
        elif search_type == "artist":
            artist_name = request.form.get("artist_name")
            songs = get_songs_by_artist(artist_name)
            if songs is None:
                error = f"No songs found for artist '{artist_name}'."
                return render_template("index.html", error=error)
            return render_template("index.html", artist_search=True, artist_name=artist_name, songs=songs)
    return render_template("index.html")

@app.route("/trends")
def trends():
    top_artists_img = get_top_artists_chart()
    top_genres_img = get_top_genres_chart()
    top_songs_list = get_top_songs_list()
    top_songs_by_artist_list = get_top_songs_by_artist_list()
    return render_template("trends.html", 
                           top_artists_img=top_artists_img, 
                           top_genres_img=top_genres_img,
                           top_songs_list=top_songs_list,
                           top_songs_by_artist_list=top_songs_by_artist_list)

@app.route("/project")
def project():
    bar_img = get_bar_graph()
    proj_scatter_img = get_project_scatter_plot()
    custom_accuracy = get_custom_accuracy()
    return render_template("project.html", bar_img=bar_img, scatter_img=proj_scatter_img,
                           custom_accuracy=custom_accuracy, mse=mse, r2=r2)

app.config["reg"] = None
if df_model is not None:
    app.config["reg"] = reg

if __name__ == '__main__':
    app.run(debug=True)
