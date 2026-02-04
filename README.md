# **End-to-End Production-Grade Machine Learning Pipeline**

This repository demonstrates a complete, production-ready Machine Learning workflow, covering everything from data ingestion to model deployment.
Although the problem itself is intentionally kept simple, the focus is on scalable architecture, clean code, modular pipelines, and industry-style project structure.

The project is fully deployed on Render and supports both training and inference pipelines.

# Key Highlights

- End-to-end ML pipeline design

- Modular, scalable production-grade folder structure

- Multiple ML models with hyperparameter tuning

- Best model selection based on evaluation metrics

- Model serialization and artifact management

- Custom logging & exception handling

- Deployed and running on Render

- Ready for extension to real-world use cases

# Workflow Overview

The project follows a complete ML lifecycle:

Data Ingestion

Load raw data

Split into train/test datasets

Store intermediate artifacts

Data Transformation

Feature engineering

Preprocessing using Scikit-Learn pipelines

Serialization of preprocessors

Model Training

Trained multiple ML models

Applied hyperparameter tuning

Compared model performance

CatBoost performed best overall

Model Evaluation

Evaluation using standard metrics

Best-performing model automatically selected

Model Serialization

Trained model and preprocessing objects saved as artifacts

Prediction Pipeline

Load saved artifacts

Perform inference on new/unseen data

Deployment

Deployed on Render

Inference served via a web interface

🗂️ Project Structure
artifacts/               # Saved models, preprocessors, and outputs
catboost_info/           # CatBoost training logs
logs/                    # Application logs
notebook/                # Experiments and EDA notebooks

src/
│
├── components/
│   ├── data_ingestion.py        # Data loading and splitting
│   ├── data_transformation.py   # Feature engineering & preprocessing
│   └── model_trainer.py         # Model training & tuning
│
├── pipelines/
│   ├── training_pipeline.py     # End-to-end training pipeline
│   └── prediction_pipeline.py   # Inference pipeline
│
├── exception.py          # Custom exception handling
├── logger.py             # Logging configuration
└── utils.py              # Common utility functions

templates/               # HTML templates for UI
app.py                   # Application entry point
requirements.txt         # Dependencies
setup.py                 # Package setup
README.md                # Project documentation

# Tech Stack

Python

Scikit-Learn

CatBoost

Pandas / NumPy

FastAPI

Render (Deployment)

Logging & Exception Handling

# Deployment

The application is deployed on Render and is fully functional for predictions.

**Note: Artifacts are loaded dynamically at runtime, following best practices for production deployments.**

# Why This Project Matters

Many ML projects stop at notebooks.
This project goes beyond modeling and focuses on:

Maintainability

Scalability

Reproducibility

Real-world deployment readiness

It reflects how ML systems are built in production, not just how models are trained.

# Future Improvements

Model versioning

Monitoring & drift detection

Dockerization

Cloud storage for artifacts
