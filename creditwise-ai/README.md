# CreditWise AI 🚀

**CreditWise AI** is a production-ready Machine Learning system for microfinance loan recommendations. It predicts creditworthiness, suggests loan amounts, and provides explainable reasoning for each decision. The system also monitors data drift in real-time and automatically retrains models when needed.

This project demonstrates a full **ML + MLOps pipeline**:

- Model training & evaluation
- API deployment (FastAPI)
- Explainable AI (SHAP)
- Drift monitoring & automated retraining
- Dashboard visualization (Streamlit)
- Docker + CI/CD deployment

## 🔹 Features

1. **Predict Loan Approval & Amount**
   - Classification: Approve / Reject
   - Regression: Recommended loan amount

2. **Explainable AI**
   - SHAP explanations for every prediction
   - JSON output and dashboard visualization

3. **Real-Time Drift Monitoring**
   - Detects feature drift using PSI and KS tests
   - Alerts when drift exceeds thresholds
   - Automatic retraining pipeline

4. **Dashboard**
   - Visualize predictions, feature importance, model performance, and drift alerts

5. **Deployment & MLOps**
   - Dockerized API + Dashboard
   - CI/CD with GitHub Actions
   - Cloud-ready deployment (Render, Railway, or GCP Cloud Run)

## 📂 Project Structure

```

creditwise-ai/
│
├── app/
│   ├── main.py           # FastAPI app for predictions
│   ├── model.py          # Model loading & prediction logic
│   ├── explain.py        # SHAP explainability logic
│   ├── schemas.py        # Pydantic schemas for input/output
│   └── utils.py          # Utility functions
│
├── training/
│   ├── preprocess.py     # Data cleaning & feature engineering
│   ├── train_classifier.py  # Train classification model
│   └── train_regressor.py   # Train regression model
│
├── monitoring/
│   ├── drift.py          # Drift detection functions
│   └── retrain.py        # Automated retraining pipeline
│
├── dashboard/
│   └── dashboard.py      # Streamlit dashboard
│
├── data/
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned & processed data
│
├── models/
│   ├── classifier_v1.pkl
│   └── regressor_v1.pkl
│
├── tests/
│   └── test_api.py       # Unit tests for API endpoints
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

```

## 🛠 Technologies Used

- **ML / AI:** Python, Scikit-learn, XGBoost, LightGBM, SHAP
- **API & Backend:** FastAPI, Pydantic
- **Monitoring & Dashboard:** Streamlit, PSI, KS Test
- **Deployment & MLOps:** Docker, GitHub Actions, Cloud deployment (Render / Railway / GCP Cloud Run)
- **Database / Logging:** SQLite / PostgreSQL

## ⚡ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/yourusername/creditwise-ai.git
cd creditwise-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start FastAPI server:

```bash
uvicorn app.main:app --reload
```

4. Run Streamlit dashboard:

```bash
streamlit run dashboard/dashboard.py
```

5. Test API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"feature1": value, "feature2": value}'
```

## 🌐 Deployment

- Build Docker image:

```bash
docker build -t creditwise-ai .
```

- Run Docker container:

```bash
docker run -p 8000:8000 creditwise-ai
```

- Use **docker-compose** for API + Dashboard stack:

```bash
docker-compose up --build
```

- CI/CD is configured via GitHub Actions to automatically build and deploy to cloud.

## 🏗 Architecture Overview

1. **Model Layer:** Trains classifier & regressor, saves artifacts, stores feature distributions
2. **API Layer:** FastAPI serves predictions & explanations, logs data
3. **Monitoring Layer:** Detects feature drift, triggers retraining
4. **Frontend Layer:** Streamlit dashboard visualizes predictions, SHAP explanations, and drift alerts
5. **Infrastructure Layer:** Docker + CI/CD pipeline, cloud-ready deployment

## 💡 Why CreditWise AI is Unique

- It combines **loan recommendation + explainable AI + drift monitoring + retraining pipeline**
- It is esigned for **real-world microfinance systems**
- Fully **production-ready & deployable**
- It includes **visibility and authority-building content** for LinkedIn / portfolio

## 🔮 Future Enhancements

- Add authentication & role-based access
- Add API rate limiting
- Multi-model ensemble for improved accuracy
- Grafana + Prometheus for production-grade monitoring
- Integrate Vertex AI / GCP services for scaling

## 👨‍💻 Author

**Bahati brenda Kizito** – Machine Learning Engineer | MLOps Enthusiast
Email: [bahatibk72@gmail.com](bahatibk72@gmail.com)
