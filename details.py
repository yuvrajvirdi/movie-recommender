
import requests
def get_movie_info(movie_name):

    URL = "http://www.omdbapi.com?t=" + movie_name.replace(" ","+") + "&apikey=" + APIKEY

    r = (requests.get(URL)).json()

    data = [r['Title'],r['Genre'], r['Plot'], r['Poster']]
    
    return data