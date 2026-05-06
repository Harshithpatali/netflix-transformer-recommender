import streamlit as st
import pandas as pd
import requests

# =====================================================
# CONFIG
# =====================================================

API_URL = "https://netflix-transformer-recommender.onrender.com/recommend"

st.set_page_config(
    page_title="Netflix Recommender",
    layout="wide"
)

# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #141414;
        color: white;
    }

    h1, h2, h3, h4 {
        color: white;
    }

    .movie-card {
        background-color: #222;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITLE
# =====================================================

st.title("🎬 Transformer Netflix Recommender")

st.markdown(
    """
    Select movies you watched and get
    transformer-based recommendations.
    """
)

# =====================================================
# LOAD MOVIES
# =====================================================

@st.cache_data
def load_movies():

    df = pd.read_csv(
        "data/movies_processed.csv"
    )

    return sorted(df["title"].tolist())

movie_titles = load_movies()

# =====================================================
# MULTISELECT
# =====================================================

selected_movies = st.multiselect(
    "Select watched movies",
    movie_titles
)

# =====================================================
# RECOMMEND BUTTON
# =====================================================

if st.button("Recommend"):

    if len(selected_movies) == 0:

        st.error(
            "Please select at least one movie"
        )

    else:

        try:

            with st.spinner(
                "Generating recommendations..."
            ):

                response = requests.post(
                    API_URL,
                    json={
                        "watched": selected_movies
                    },
                    timeout=120
                )

                # =====================================
                # CHECK RESPONSE
                # =====================================

                response.raise_for_status()

                data = response.json()

                recommendations = data.get(
                    "recommendations",
                    []
                )

                # =====================================
                # EMPTY RESULTS
                # =====================================

                if len(recommendations) == 0:

                    st.warning(
                        "No recommendations found."
                    )

                else:

                    st.subheader(
                        "🎯 Recommended Movies"
                    )

                    cols = st.columns(5)

                    for idx, rec in enumerate(
                        recommendations
                    ):

                        with cols[idx % 5]:

                            st.markdown(
                                f"""
                                <div class="movie-card">
                                    <h4>
                                        🎬 {rec['title']}
                                    </h4>
                                    <p>
                                        Score:
                                        {rec['score']:.4f}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        # =============================================
        # REQUEST ERRORS
        # =============================================

        except requests.exceptions.Timeout:

            st.error(
                "Request timed out. "
                "Render free tier may be waking up."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to backend API."
            )

        except requests.exceptions.HTTPError as e:

            st.error(
                f"HTTP Error: {e}"
            )

        except Exception as e:

            st.error(
                f"Unexpected Error: {str(e)}"
            )