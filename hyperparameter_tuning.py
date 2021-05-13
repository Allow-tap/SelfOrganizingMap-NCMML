import ast
import pickle

import matplotlib.pyplot as plt
from minisom import MiniSom
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.model_selection import ParameterGrid

# Adjust these parameters
som_neurons = (44, 44)
epochs = 100
learning_rates = [0.001, 0.01, 0.1]
sigmas = [0.5, 1, 1.5]
genres = ["rock",
          "classical",
          "latin",
          "pop",
          "jazz",
          "soul",
          "classic bollywood",
          "rap",
          "fold",
          "funk",
          "opera"]
features = ['acousticness', 'instrumentalness', 'loudness', 'energy',
            'danceability', 'valence']
cluster_method = AgglomerativeClustering(n_clusters=len(genres))
# cluster_method = MiniBatchKMeans(n_clusters=len(genres))

# Leave the rest
print('Reading data...')
tracks = pd.read_csv('data/tracks_with_genres.csv')
tracks['genres'] = tracks['genres'].apply(ast.literal_eval)
tracks = tracks.explode('genres')
tracks.rename(columns={'genres': 'genre'}, inplace=True)
tracks_subset = tracks[tracks['genre'].isin(genres)]

print('Transforming data...')
tracks_values = tracks_subset[features].values
tracks_values = (tracks_values - tracks_values.mean(axis=0))\
                / tracks_values.std(axis=0)

rng = np.random.default_rng()
rng.shuffle(tracks_values, axis=0)

parameter_combinations = list(ParameterGrid({
    'learning_rate': learning_rates,
    'sigma': sigmas}))

print('Training models...')
for i, params in enumerate(parameter_combinations):
    learning_rate, sigma = params['learning_rate'], params['sigma']
    print(f'Running model {i+1} of {len(parameter_combinations)} with '
          f'learning rate={learning_rate} and sigma={sigma}')
    som = MiniSom(som_neurons[0], som_neurons[1], len(features),
                  sigma=sigma, learning_rate=learning_rate)

    quantization_errors = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for i in range(epochs):
        print(f'Running epoch {i+1}/{epochs}')
        som.train(tracks_values, tracks_values.shape[0], verbose=True)
        quantization_errors.append(som.quantization_error(tracks_values))

        som_weights = som.get_weights()
        som_weights_flat = som_weights.reshape(som_neurons[0] * som_neurons[1],
                                               len(features))
        labels = cluster_method.fit_predict(som_weights_flat)
        silhouette = silhouette_score(som_weights_flat, labels)
        davies_bouldin = davies_bouldin_score(som_weights_flat, labels)
        calinski_harabasz = calinski_harabasz_score(som_weights_flat, labels)

        print(f'Silhouette score: {silhouette:.3f}')
        print(f'Davies-Bouldin score: {davies_bouldin:.3f} (lower is better)')
        print(f'Calinski-Harabasz score: {calinski_harabasz:.3f}'
              f'(higher is better)\n')

        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        calinski_harabasz_scores.append(calinski_harabasz)

    fig, ax1 = plt.subplots()
    ax1.plot(silhouette_scores, color='b', linestyle='-', label='Silhouette')
    ax1.plot(davies_bouldin_scores, color='b', linestyle='-.',
             label='Davies-Bouldin')
    ax1.plot(quantization_errors, color='b', linestyle='--',
             label='Quantization error')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Epochs')

    ax2 = plt.twinx()
    ax2.plot(calinski_harabasz_scores, color='r', linestyle=':',
             label='Calinski-Harabasz')
    ax2.tick_params(axis='y', labelcolor='r')
    ax1.legend()
    ax2.legend()
    fig.tight_layout()

    cluster_method_name = type(cluster_method).__name__

    plt.savefig(f'figures/{cluster_method_name}_training_'
                f'lr_{learning_rate}_sigma_{sigma}.png')
    plt.show()

    with open(f'models/{cluster_method_name}_training_'
              f'lr_{learning_rate}_sigma_{sigma}.p', 'wb') as model_file:
        pickle.dump(som, model_file)
