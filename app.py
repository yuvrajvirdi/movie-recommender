from flask import Flask, render_template, request
from recommender import get_recommendations, cosine_sim2
from details import get_movie_info


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/get_movies', methods=['GET', 'POST'])
def get_movies():
    if request.method == "POST":

        movie = request.form.get('movie')

        filter = request.form.get('filter')

        if filter == "content":
            movies = get_recommendations(movie)
        else:
            movies = get_recommendations(movie, cosine_sim2)

        movies_dict = {}
        for i in range(len(movies)):
            movies_dict[movies[i]] = get_movie_info(movies[i])

        return render_template("movies.html", movies_dict=movies_dict, movie=movie, movies=movies, length=len(movies))

    else:
        return render_template("index.html")

@app.errorhandler(500)
def handle_500(e):
    error = 'That movie does not exist in the database, try again'
    return render_template('index.html', error=error)

if __name__ == "__main__":
    app.run()