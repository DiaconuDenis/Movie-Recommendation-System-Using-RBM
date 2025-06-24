# Movie-Recommendation-System-Using-RBM

Project Overview
This project implements a Movie Recommendation System leveraging a Restricted Boltzmann Machine (RBM), developed from scratch using PyTorch. The system is designed to provide personalized movie recommendations by learning complex patterns and latent features from user-movie interaction data.

The RBM, a type of generative stochastic artificial neural network, is particularly well-suited for collaborative filtering tasks, where the goal is to predict a user's preference for items they haven't yet rated, based on their past ratings and the ratings of similar users.

How It Works: RBMs for Recommendation
A Restricted Boltzmann Machine (RBM) is a two-layer neural network (visible and hidden layers) with no intra-layer connections. For recommendation systems, RBMs learn to represent user preferences:

Visible Layer: Represents the user's ratings for movies. Each neuron in the visible layer corresponds to a movie, and its state (e.g., binary for watched/not watched, or actual rating values) reflects the user's interaction with that movie.
Hidden Layer: Captures abstract features or latent factors that explain the observed ratings. These hidden units can represent movie genres, thematic elements, or user taste profiles that aren't explicitly given in the data.
Learning Process: The RBM is trained using an algorithm like Contrastive Divergence (CD) to adjust the weights and biases between the visible and hidden layers. This process allows the RBM to learn a joint probability distribution over the visible and hidden units, effectively learning to reconstruct user ratings.
Once trained, the RBM can predict a user's rating for unrated movies by inferring the hidden unit states from their known ratings and then using these hidden states to reconstruct the ratings for all movies, including the unrated ones.

Dataset
This project utilizes the MovieLens dataset.This dataset consists of movie ratings provided by users, typically including:

User IDs
Movie IDs
Ratings (e.g., 1-5 stars)
(Optional) Timestamps
The data undergoes pre-processing to transform it into a format suitable for the RBM, often a user-movie interaction matrix where missing values represent unrated movies.

Key Features
PyTorch Implementation: The RBM model is built from the ground up in PyTorch, providing flexibility and a deep understanding of its internal workings.
Contrastive Divergence (CD) Training: Implements the CD algorithm for efficient training of the RBM.
Personalized Recommendations: Generates a ranked list of recommended movies for a given user based on their predicted ratings.
Scalability: While a basic implementation, the principles can be extended for larger datasets and more complex RBM architectures.
