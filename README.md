# Building a Recommendation System from Scratch in PyTorch

A deep dive into collaborative filtering, this project documents the process of building a movie recommendation engine from the ground up using PyTorch to predict user ratings. It covers the entire machine learning lifecycle, from data exploration to iteratively diagnosing and solving complex model performance challenges.

---

## Project Overview

The goal of this project was to build a personalized movie recommendation system using the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/). Rather than relying on high-level recommendation libraries, the core **Matrix Factorization** model was implemented from scratch to gain a fundamental understanding of collaborative filtering, model architecture, and the training process.

The final model is capable of predicting a user's rating for a movie they haven't seen, achieving a **Validation Root Mean Squared Error (RMSE) of 1.05**. The project also will include functionality to generate personalized recommendations for a new user after learning their tastes from a small sample of their ratings.

---

## Key Features

* **Custom PyTorch Model:** A Matrix Factorization model built from the ground up using `nn.Module` and `nn.Embedding` layers.
* **Iterative Debugging:** A documented process of diagnosing and solving real-world ML problems, including unconstrained model outputs and overfitting.
* **Hyperparameter Tuning:** Systematic experiments to find the optimal learning rate and model complexity, drastically improving performance.
* **New User Recommendations:** A solution to the "cold start" problem, allowing a new user to provide their own ratings and receive personalized recommendations.
* **Comprehensive EDA:** Initial data analysis to identify key characteristics like popularity bias and the long-tail distribution of ratings.

---

## Tech Stack

* **Primary Language:** Python
* **Machine Learning:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Development Environment:** Jupyter Notebook

---

## Project Narrative & Key Learnings

The core of this project was not just building a model, but the iterative process of refining it.

1.  **Baseline & Architecture Correction:** An initial model produced nonsensical results due to an unconstrained output. The first major step was diagnosing this issue and re-architecting the model's `forward` pass with a `Sigmoid` activation function to correctly map predictions to the 1-5 star rating scale.

2.  **Tuning for Performance:** With a stable architecture, a series of experiments were run to optimize performance. A learning rate of `0.01` was identified as optimal.

3.  **Combating Overfitting:** By plotting the training and validation loss curves, a clear case of overfitting was identified. This was mitigated by introducing `weight_decay` to the optimizer and using the learning curves to determine the optimal number of training epochs (an informal method of "early stopping").

4.  **Scaling Up:** The model's final performance boost came from increasing its complexity (`embedding_dim`) and the amount of data it was trained on, demonstrating the trade-off between model capacity and data volume.

This journey highlights a realistic machine learning workflow: start simple, identify the biggest problem, solve it systematically, and repeat.

---

## Future Work

This project serves as a strong foundation for several potential enhancements:

* **Implement Bias Terms:** Add user-specific and movie-specific bias terms to the model, a standard technique that could further improve the RMSE.
* **Build a Hybrid Model:** Incorporate content-based features (like movie `genres`) into the model to improve recommendations for users or items with few interactions.
* **Deploy as an API:** Wrap the trained model in a simple API using a framework like FastAPI to serve recommendations in real-time.