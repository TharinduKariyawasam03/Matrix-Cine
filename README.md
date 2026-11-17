# MATRIX CINE

**Discover. Match. Watch.**

MATRIX CINE is a cinematic Streamlit application that recommends movies similar to a title in the supplied TMDB dataset. It uses the precomputed movie data and similarity matrix in `movie_recommendation_model.pkl`; it does not retrain or replace the saved model.

## Features

- Search the movie library and request the top 10 similar titles.
- View available release years, ratings, and movie overviews.
- See each recommendation's rank and saved similarity score.
- Keep a short recently explored list during the current session.
- Handle missing metadata and invalid or incompatible model files gracefully.

## Technologies

- Python
- Streamlit
- pandas and NumPy
- joblib and scikit-learn (for loading the saved artifact)

## Machine-learning methodology

The notebook prepares TMDB movie metadata, applies a TF-IDF vectorizer to movie overviews, and calculates a precomputed kernel similarity matrix. The application uses the matrix and movie rows as saved, sorts scores in descending order, excludes the selected movie, and returns up to 10 results. It does not retrain the vectorizer or calculate a replacement matrix.

**Artifact note:** The supplied model key is named `cosine_sim`, but the notebook's demonstrated workflow creates its saved score matrix with `sigmoid_kernel`. MATRIX CINE preserves and displays the saved values without changing that workflow. Scores are shown as percentages by multiplying the saved scores by 100; they are not probabilities of liking or watching a movie.

The notebook and artifact contain 481 movie rows. The app checks that the matrix dimensions and saved title-to-row mapping agree before serving recommendations; it uses matrix row positions rather than the DataFrame's original (sparse) row labels.

## Run locally

1. Place `app.py`, `movie_recommendation_model.pkl`, and `requirements.txt` in the same directory.
2. From that directory, create and activate a virtual environment if desired.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the application:

   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Community Cloud

1. Push `app.py`, `movie_recommendation_model.pkl`, `requirements.txt`, and this README to a GitHub repository. The model file must be committed with the app so the deployment can load it.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and select **Create app**.
3. Choose the repository, branch, and `app.py` as the main file.
4. Deploy. Community Cloud installs the packages from `requirements.txt`.

The supplied model is approximately 10 MB, which is suitable for a standard GitHub repository.
