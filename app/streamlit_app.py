import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Netflix Recommender",
    layout="wide"
)

st.title("🎬 Transformer Netflix Recommender")

movies_df = pd.read_csv(
    "data/movies_processed.csv"
)

movie_titles = sorted(
    movies_df["title"].tolist()
)

selected_movies = st.multiselect(
    "Select watched movies",
    movie_titles
)

if st.button("Recommend"):

    if len(selected_movies) == 0:

        st.error(
            "Please select movies"
        )

    else:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/recommend",
                json={
                    "watched": selected_movies
                }
            )

            data = response.json()

            recommendations = data[
                "recommendations"
            ]

            st.subheader("Recommendations")

            cols = st.columns(5)

            for idx, rec in enumerate(recommendations):

                with cols[idx % 5]:

                    st.markdown(
                        f"""
                        ### 🎬 {rec['title']}
                        Score: {rec['score']:.4f}
                        """
                    )

        except Exception as e:

            st.error(str(e))