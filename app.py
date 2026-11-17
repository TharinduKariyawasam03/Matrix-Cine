from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "movie_recommendation_model.pkl"
HISTORY_LIMIT = 5


@dataclass(frozen=True)
class RecommendationModel:
    data: pd.DataFrame
    similarity: np.ndarray
    title_to_row: dict[str, int]
    titles: tuple[str, ...]


@st.cache_resource(show_spinner="Loading the movie library...")
def load_recommendation_model(model_path: str) -> RecommendationModel:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise ValueError("The model artifact must be a dictionary.")

    data = artifact.get("data")
    if not isinstance(data, pd.DataFrame) or "original_title" not in data.columns:
        raise ValueError("The model artifact does not contain valid movie data.")

    similarity = np.asarray(artifact.get("cosine_sim"))
    movie_count = len(data)
    if similarity.ndim != 2 or similarity.shape != (movie_count, movie_count):
        raise ValueError("The similarity matrix dimensions do not match the movie data.")
    if not np.isfinite(similarity).all():
        raise ValueError("The similarity matrix contains invalid scores.")

    titles_by_row: list[str] = []
    title_to_row: dict[str, int] = {}
    for row_number, raw_title in enumerate(data["original_title"].tolist()):
        if pd.isna(raw_title):
            continue
        title = str(raw_title).strip()
        if not title:
            continue
        if title in title_to_row:
            raise ValueError("The movie data contains duplicate titles.")
        titles_by_row.append(title)
        title_to_row[title] = row_number

    if not title_to_row:
        raise ValueError("The movie data does not contain searchable titles.")

    # Saved index values are matrix row positions; DataFrame labels may be sparse.
    saved_indices = artifact.get("indices")
    if isinstance(saved_indices, pd.Series):
        if len(saved_indices) != movie_count:
            raise ValueError("The saved title index does not match the movie data.")
        positions: set[int] = set()
        for title, raw_position in saved_indices.items():
            if not isinstance(title, str) or not isinstance(raw_position, (int, np.integer)):
                raise ValueError("The saved title index has an invalid entry.")
            position = int(raw_position)
            if position < 0 or position >= movie_count or position in positions:
                raise ValueError("The saved title index has an invalid row position.")
            if str(data.iloc[position]["original_title"]) != title:
                raise ValueError("The saved title index does not match the movie row order.")
            positions.add(position)
        if positions != set(range(movie_count)):
            raise ValueError("The saved title index does not cover every movie row.")
        title_to_row = {str(title): int(position) for title, position in saved_indices.items()}

    return RecommendationModel(
        data=data,
        similarity=similarity,
        title_to_row=title_to_row,
        titles=tuple(title_to_row),
    )


def get_recommendations(
    model: RecommendationModel, title: str, limit: int = 10
) -> list[tuple[int, float]]:
    if title not in model.title_to_row:
        raise ValueError("Select a movie from the available library.")

    selected_row = model.title_to_row[title]
    scores = model.similarity[selected_row]
    ranked_rows = sorted(
        (row for row in range(len(model.data)) if row != selected_row),
        key=lambda row: scores[row],
        reverse=True,
    )
    return [(row, float(scores[row])) for row in ranked_rows[:limit]]


def get_movie_year(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(parsed) else int(parsed.year)


def get_movie_rating(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if np.isfinite(rating) else None


st.set_page_config(
    page_title="MATRIX CINE | Discover. Match. Watch.",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    movie_model = load_recommendation_model(str(MODEL_PATH))
except Exception:
    LOGGER.exception("Unable to load the MATRIX CINE recommendation model.")
    st.error(
        "The movie recommendation model could not be loaded. "
        "Check that the model file is present and compatible with the listed dependencies."
    )
    st.stop()

if "recent_movies" not in st.session_state:
    st.session_state.recent_movies = []
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "search_version" not in st.session_state:
    st.session_state.search_version = 0

st.markdown(
    """
    <style>
    :root {
        --matrix-navy: #0B1D38;
        --matrix-dark: #07152A;
        --matrix-orange: #FF8A00;
        --matrix-white: #F5F7FA;
        --matrix-slate: #A7B4C7;
        --matrix-card: #142B49;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 80% 0%, rgba(255, 138, 0, .10), transparent 34%),
            linear-gradient(145deg, #07152A 0%, #0B1D38 56%, #09182D 100%);
        color: var(--matrix-white);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07152A 0%, #0B1D38 100%);
        border-right: 1px solid rgba(148, 163, 184, .14);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--matrix-slate);
    }
    .block-container { max-width: 1440px; padding-top: 2rem; }
    .brand-mark {
        color: var(--matrix-white);
        font-size: 1.4rem;
        font-weight: 850;
        letter-spacing: .12em;
        margin: .25rem 0 0;
    }
    .brand-mark span, .accent { color: var(--matrix-orange); }
    .tagline {
        color: #A7B4C7;
        font-size: .88rem;
        letter-spacing: .09em;
        margin-top: .2rem;
    }
    .eyebrow {
        color: var(--matrix-orange);
        font-size: .75rem;
        font-weight: 750;
        letter-spacing: .18em;
        text-transform: uppercase;
    }
    .welcome-panel {
        border: 1px solid rgba(255, 138, 0, .22);
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(20, 43, 73, .94), rgba(11, 29, 56, .78));
        padding: clamp(1.5rem, 4vw, 3rem);
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 55px rgba(0, 0, 0, .16);
    }
    .welcome-panel h1 {
        color: var(--matrix-white);
        font-size: clamp(2rem, 5vw, 3.4rem);
        line-height: 1.1;
        margin: .45rem 0 .7rem;
    }
    .welcome-panel p { color: #B5C0D0; margin-bottom: 0; }
    .movie-panel {
        background: rgba(20, 43, 73, .78);
        border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin: 1rem 0 1.6rem;
    }
    .movie-panel h2 { color: var(--matrix-white); margin: .35rem 0 .65rem; }
    .movie-meta { color: #B5C0D0; font-size: .92rem; margin-bottom: .85rem; }
    .movie-overview { color: #D5DCE6; line-height: 1.65; }
    .recommendation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 255px), 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .recommendation-card {
        background: linear-gradient(145deg, #142B49, #10243E);
        border: 1px solid rgba(148, 163, 184, .14);
        border-radius: 15px;
        min-height: 150px;
        padding: 1.1rem;
        transition: transform .18s ease, border-color .18s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 138, 0, .55);
    }
    .rank-line { display: flex; justify-content: space-between; align-items: center; }
    .rank-number {
        color: var(--matrix-orange);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
    }
    .similarity {
        color: #FFD09A;
        font-size: .82rem;
        font-weight: 700;
    }
    .recommendation-card h3 {
        color: var(--matrix-white);
        font-size: 1.08rem;
        line-height: 1.35;
        margin: 1rem 0 .55rem;
    }
    .card-meta { color: #A7B4C7; font-size: .85rem; }
    [data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.75rem;
    }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stButton"] button[data-testid="stBaseButton-primary"] {
        background: var(--matrix-orange);
        border-color: var(--matrix-orange);
        color: #07152A !important;
        --text-color: #07152A;
        font-weight: 800;
    }
    [data-testid="stButton"] button[kind="primary"] *,
    [data-testid="stButton"] button[data-testid="stBaseButton-primary"] * {
        color: #07152A !important;
    }
    [data-testid="stButton"] button[kind="primary"]:hover,
    [data-testid="stButton"] button[data-testid="stBaseButton-primary"]:hover {
        background: #FF9D2E;
        border-color: #FF9D2E;
        color: #07152A !important;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .welcome-panel { border-radius: 15px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand-mark">MATRIX <span>CINE</span></div>'
        '<div class="tagline">Discover. Match. Watch.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Find your next film")
    selected_search = st.selectbox(
        "Movie title",
        options=movie_model.titles,
        key=f"movie_search_{st.session_state.search_version}",
        label_visibility="collapsed",
    )
    find_column, clear_column = st.columns(2)
    with find_column:
        find_clicked = st.button("Find Similar", type="primary", use_container_width=True)
    with clear_column:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.selected_movie = None
        st.session_state.search_version += 1
        st.rerun()
    if find_clicked:
        if selected_search in movie_model.title_to_row:
            st.session_state.selected_movie = selected_search
            recent = st.session_state.recent_movies
            st.session_state.recent_movies = [selected_search] + [
                title for title in recent if title != selected_search
            ]
            st.session_state.recent_movies = st.session_state.recent_movies[:HISTORY_LIMIT]
        else:
            st.error("Choose a movie from the available list.")

    st.markdown("---")
    st.markdown("#### Recently explored")
    if st.session_state.recent_movies:
        for recent_title in st.session_state.recent_movies:
            st.markdown(
                f'<div class="tagline" style="padding:.3rem 0;">'
                f'<span class="accent">›</span> {html.escape(recent_title)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Your explored titles will appear here.")
    st.markdown("---")
    st.caption(f"{len(movie_model.titles):,} movies in the library")

st.markdown(
    """
    <section class="welcome-panel">
      <div class="eyebrow">Intelligent Movie Recommendation System</div>
      <h1>Discover your next<br><span class="accent">great watch.</span></h1>
      <p>Explore films that match your taste, powered by the MATRIX CINE recommendation model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

selected_movie = st.session_state.selected_movie
if selected_movie is None:
    st.info("Choose a movie in the sidebar and select **Find Similar** to explore recommendations.")
elif selected_movie not in movie_model.title_to_row:
    st.error("That movie is not available in the recommendation library. Select another title.")
else:
    movie_row = movie_model.title_to_row[selected_movie]
    movie = movie_model.data.iloc[movie_row]
    year = get_movie_year(movie.get("release_date"))
    rating = get_movie_rating(movie.get("vote_average"))
    overview = movie.get("overview")
    if overview is None or pd.isna(overview) or not str(overview).strip():
        overview_text = "No overview available."
    else:
        overview_text = html.escape(str(overview).strip())

    details = []
    if year is not None:
        details.append(str(year))
    if rating is not None:
        details.append(f"★ {rating:.1f} / 10")
    metadata = " &nbsp;·&nbsp; ".join(details) if details else "Movie details unavailable"

    st.markdown(
        f'<section class="movie-panel">'
        f'<div class="eyebrow">Your selection</div>'
        f'<h2>{html.escape(selected_movie)}</h2>'
        f'<div class="movie-meta">{metadata}</div>'
        f'<div class="movie-overview">{overview_text}</div>'
        f'</section>',
        unsafe_allow_html=True,
    )

    recommendations = get_recommendations(movie_model, selected_movie, limit=10)
    st.subheader("Top 10 similar movies")
    if not recommendations:
        st.info("No similar movies are available for this title.")
    else:
        cards = []
        for rank, (row_number, score) in enumerate(recommendations, start=1):
            recommendation = movie_model.data.iloc[row_number]
            recommendation_title = html.escape(str(recommendation["original_title"]))
            recommendation_year = get_movie_year(recommendation.get("release_date"))
            recommendation_rating = get_movie_rating(recommendation.get("vote_average"))
            card_details = []
            if recommendation_year is not None:
                card_details.append(str(recommendation_year))
            if recommendation_rating is not None:
                card_details.append(f"★ {recommendation_rating:.1f} / 10")
            card_metadata = " &nbsp;·&nbsp; ".join(card_details) or "Details unavailable"
            percentage = min(max(score, 0.0), 1.0) * 100
            cards.append(
                f'<article class="recommendation-card">'
                f'<div class="rank-line"><span class="rank-number">MATCH {rank:02d}</span>'
                f'<span class="similarity">{percentage:.1f}%</span></div>'
                f'<h3>{recommendation_title}</h3>'
                f'<div class="card-meta">{card_metadata}</div>'
                f'</article>'
            )
        st.markdown(
            '<div class="recommendation-grid">' + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Similarity percentages are the saved model scores scaled by 100, not viewing probabilities.")
