# 🎬 Transformer-Based Netflix Recommendation System

A production-style movie recommendation system built using Transformer architecture, hybrid filtering, FastAPI, and Streamlit.

This project predicts the next movies a user may like based on their watched movie sequence using a Transformer-based sequential recommender system inspired by modern recommendation architectures used in companies like Netflix, YouTube, and Amazon.
live link : https://netflix-transformer-recommender-rsjgta5q2bvzgaqfappzvwf.streamlit.app/
---

# 🚀 Features

## ✅ Transformer-Based Sequential Recommendation

- Learns user watching patterns
- Captures temporal movie behavior
- Uses self-attention mechanism
- Predicts next likely movie

---

## ✅ Attention Visualization

- Extracts Transformer attention weights
- Helps explain recommendation decisions
- Visualized using heatmaps

---

## ✅ Hybrid Recommendation

Combines:

- Transformer sequential scores
- Content-based genre similarity

Hybrid scoring:

```python
final_score = (
    alpha * transformer_score
    +
    (1 - alpha) * content_score
)
```

---

## ✅ Ranking Model

Uses:

- LightGBM LambdaRank
- Transformer score
- Popularity
- Genre similarity
- Sequence features

to improve recommendation ordering.

---

## ✅ FastAPI Backend

Production-style inference API:

- low-latency serving
- model caching
- REST endpoints
- scalable deployment

---

## ✅ Streamlit Frontend

Netflix-style UI:

- movie selection
- recommendation display
- cloud deployment ready

---

# 🧠 Why Transformer-Based Recommendation?

Traditional recommender systems like:

- Collaborative Filtering
- Matrix Factorization
- SVD
- KNN

mainly learn static relationships between users and items.

They struggle with:

- changing user interests
- sequential behavior
- session-based recommendations
- temporal patterns

---

# 🔥 Why Transformers Are Better

Transformers learn:

- order of watched movies
- long-range dependencies
- dynamic user behavior
- contextual movie relationships

Example:

A traditional model sees:

```text
User likes action movies
```

A Transformer sees:

```text
User watched:
Batman → Joker → Inception

Next likely interest:
Psychological thriller
```

This sequence-awareness makes Transformers significantly more powerful for modern recommendation systems.

---

# ⚡ Advantages Over Traditional Models

| Traditional Models | Transformer Models |
|---|---|
| Static preferences | Dynamic sequential behavior |
| Weak temporal learning | Strong temporal learning |
| Limited context | Full sequence context |
| Manual feature engineering | Learns representations automatically |
| Poor session recommendation | Excellent session recommendation |
| Limited personalization | Deep personalization |

---

# 🏗️ Architecture

```text
User Movie Sequence
        ↓
Embedding Layer
        ↓
Positional Encoding
        ↓
Multi-Head Self Attention
        ↓
Transformer Representation
        ↓
Next Movie Prediction
        ↓
Hybrid Scoring
        ↓
Ranking Model
        ↓
Top-N Recommendations
```

---

# 📂 Project Structure

```text
netflix-transformer-recommender/
│
├── app/
│   ├── api.py
│   ├── streamlit_app.py
│
├── src/
│   ├── data/
│   ├── models/
│   ├── hybrid/
│   ├── ranking/
│   ├── utils/
│   ├── train.py
│   ├── inference.py
│
├── saved_models/
│
├── requirements.txt
├── README.md
```

---

# 📦 Dataset

Uses MovieLens dataset:

- ratings.csv
- movies.csv

Dataset contains:

- user ratings
- movie metadata
- timestamps
- genres

---

# ⚙️ Training Pipeline

## 1. Data Preprocessing

- sort by timestamp
- encode movie IDs
- filter active users
- filter popular movies

---

## 2. Sequence Generation

Creates:

```text
[input movie sequence] → next movie
```

Example:

```text
Toy Story → Heat → Batman
                ↓
           predict next movie
```

---

## 3. Transformer Training

- CrossEntropyLoss
- Adam optimizer
- Gradient clipping
- Validation monitoring
- Early stopping

---

# 🚀 Running Locally

## Create virtual environment

```bash
python -m venv venv
```

---

## Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ⚙️ Data Processing

```bash
python -m src.data.preprocess
python -m src.data.sequence_builder
```

---

# 🧠 Train Model

```bash
python -m src.train
```

---

# 🚀 Run Backend

```bash
uvicorn app.api:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

---

# 🎬 Run Frontend

```bash
streamlit run app/streamlit_app.py
```

Frontend:

```text
http://localhost:8501
```

---

# ☁️ Deployment

## Backend

Deployed on:

- Render

## Frontend

Deployed on:

- Streamlit Cloud

---

# 🧠 Technologies Used

- Python
- PyTorch
- FastAPI
- Streamlit
- LightGBM
- Pandas
- NumPy
- Scikit-learn

---

# 📈 Future Improvements

- SASRec architecture
- BERT4Rec
- Candidate retrieval stage
- Approximate nearest neighbor search
- Redis caching
- Docker + Kubernetes deployment
- GPU inference optimization
- Real-time recommendation streaming

---

# 🎯 Learning Outcomes

This project demonstrates:

- Transformer architectures
- Sequential recommendation systems
- Recommendation ranking
- Production ML pipelines
- FastAPI deployment
- Streamlit deployment
- Hybrid recommender systems
- Attention mechanisms

---

# 👨‍💻 Author

Harshith

Master's in Applied Mathematics and Computing

Interested in:

- Machine Learning
- AI Systems
- Recommendation Systems
- Deep Learning
- MLOps
