import os
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from langchain.tools import tool
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# --- Spotify Tool Setup ---
try:
    spotify = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        )
    )
except Exception as e:
    print(f"Could not initialize Spotify client: {e}")
    spotify = None

# --- TMDB Tool ---
@tool
def fetch_movie_details(movie_title: str) -> dict:
    """
    Fetches detailed information for a specific movie title from The Movie Database (TMDB).
    Use this to get movie details like overview, poster path, release date, genres, etc.
    """
    api_key = os.getenv("TMDB_API_KEY")
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    
    try:
        search_response = requests.get(search_url).json()
        if not search_response.get("results"):
            return {"error": f"Movie '{movie_title}' not found."}

        movie_id = search_response["results"][0]["id"]
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits"
        details_response = requests.get(details_url).json()

        cast = [actor['name'] for actor in details_response.get('credits', {}).get('cast', [])[:5]]
        director = next((crew['name'] for crew in details_response.get('credits', {}).get('crew', []) if crew['job'] == 'Director'), 'N/A')

        return {
            "id": details_response.get("id"),
            "title": details_response.get("title"),
            "overview": details_response.get("overview"),
            "posterPath": f"https://image.tmdb.org/t/p/w500{details_response.get('poster_path')}" if details_response.get('poster_path') else None,
            "releaseDate": details_response.get("release_date"),
            "voteAverage": details_response.get("vote_average"),
            "genres": details_response.get("genres", []),
            "runtime": details_response.get("runtime"),
            "director": director,
            "cast": cast,
        }
    except Exception as e:
        return {"error": f"An error occurred while fetching movie details for '{movie_title}': {str(e)}"}

# --- Google Books Tool ---
@tool
def fetch_book_details(book_title: str) -> dict:
    """
    Fetches detailed information for a specific book title from the Google Books API.
    Use this to get book details like authors, description, page count, cover image, etc.
    """
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_title}&key={api_key}"
    
    try:
        response = requests.get(url).json()
        if not response.get("items"):
            return {"error": f"Book '{book_title}' not found."}

        book_info = response["items"][0]["volumeInfo"]
        
        return {
            "id": response["items"][0].get("id"),
            "title": book_info.get("title"),
            "authors": book_info.get("authors", []),
            "description": book_info.get("description"),
            "pageCount": book_info.get("pageCount"),
            "categories": book_info.get("categories", []),
            "imageLinks": book_info.get("imageLinks", {}),
            "previewLink": book_info.get("previewLink"),
            "publishedDate": book_info.get("publishedDate"),
        }
    except Exception as e:
        return {"error": f"An error occurred while fetching book details for '{book_title}': {str(e)}"}

# --- Spotify Tool ---
@tool
def fetch_song_details(song_title_and_artist: str) -> dict:
    """
    Fetches detailed information for a specific song from the Spotify API.
    Input should be a string in the format 'Song Title by Artist Name'.
    Use this to get song details like album, release date, Spotify URL, album art, etc.
    """
    if not spotify:
        return {"error": "Spotify client not initialized. Check credentials."}

    try:
        results = spotify.search(q=song_title_and_artist, type="track", limit=1)
        if not results or not results["tracks"]["items"]:
            return {"error": f"Song '{song_title_and_artist}' not found."}

        track = results["tracks"]["items"][0]
        
        return {
            "id": track.get("id"),
            "title": track.get("name"),
            "artist": ", ".join([artist["name"] for artist in track["artists"]]),
            "album": track.get("album", {}).get("name"),
            "releaseDate": track.get("album", {}).get("release_date"),
            "duration": track.get("duration_ms"),
            "previewUrl": track.get("preview_url"),
            "spotifyUrl": track.get("external_urls", {}).get("spotify"),
            "albumArt": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
            "popularity": track.get("popularity"),
        }
    except Exception as e:
        return {"error": f"An error occurred while fetching song details for '{song_title_and_artist}': {str(e)}"}