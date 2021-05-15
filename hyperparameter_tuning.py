import ast
import multiprocessing
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
sigmas = [1.5, 2, 2.5, 3, 3.5]

# Set this manually if you don't want to use all CPUs
# max_n_cpus = 4  # like this
max_n_cpus = multiprocessing.cpu_count()

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

for i, x in enumerate(parameter_combinations):
    parameter_combinations[i]['index'] = i+1


def evaluate_model(params):
    learning_rate, sigma = params['learning_rate'], params['sigma']
    index = params['index']

    print(f'Running model {index} with learning rate={learning_rate} and '
          f'sigma={sigma}')
    som = MiniSom(som_neurons[0], som_neurons[1], len(features),
                  sigma=sigma, learning_rate=learning_rate)

    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for epoch in range(epochs):
        print(f'Model {index} running epoch {epoch+1}/{epochs}')
        som.train(tracks_values, tracks_values.shape[0], verbose=False)

        som_weights = som.get_weights()
        som_weights_flat = som_weights.reshape(som_neurons[0] * som_neurons[1],
                                               len(features))
        labels = cluster_method.fit_predict(som_weights_flat)
        silhouette = silhouette_score(som_weights_flat, labels)
        davies_bouldin = davies_bouldin_score(som_weights_flat, labels)
        calinski_harabasz = calinski_harabasz_score(som_weights_flat, labels)

        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        calinski_harabasz_scores.append(calinski_harabasz)

        fig, ax1 = plt.subplots()
        x_range = range(1, len(silhouette_scores)+1)
        ax1.plot(x_range, silhouette_scores, color='b',
                 linestyle='-', label='Silhouette')
        ax1.plot(x_range, davies_bouldin_scores, color='b',
                 linestyle='-.', label='Davies-Bouldin')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_xlabel('Epochs')

        ax2 = plt.twinx()
        ax2.plot(x_range, calinski_harabasz_scores, color='r',
                 linestyle=':', label='Calinski-Harabasz')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(0, 1000)
        ax1.legend()
        ax2.legend()
        fig.tight_layout()

        cluster_method_name = type(cluster_method).__name__

        plt.savefig(f'figures/{cluster_method_name}_training_'
                    f'lr_{learning_rate}_sigma_{sigma}.png')
        plt.close()

        with open(f'models/{cluster_method_name}_training_'
                  f'lr_{learning_rate}_sigma_{sigma}.p', 'wb') as model_file:
            pickle.dump(som, model_file)

        print(f'Model {index} done with epoch {epoch+1}.')

    print(f'Model {index} with learning rate={learning_rate} and '
          f'sigma={sigma} finished.')


print('Training models...')
pool = multiprocessing.Pool(min(multiprocessing.cpu_count(),
                                len(parameter_combinations)))
pool.map(evaluate_model, parameter_combinations)
