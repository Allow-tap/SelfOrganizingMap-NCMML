import ast
import itertools
import multiprocessing
import pickle

from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from minisom import MiniSom
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

# Adjust these parameters
som_neurons = (38, 38)
epochs = [10, 40]
learning_rates = [[0.1, 0], [0.01, 0]]
sigmas = [[37, 1], [10, 1]]
min_features = 2
max_features = 6

# Set this manually if you don't want to use all CPUs
# max_n_cpus = 4  # like this
max_n_cpus = multiprocessing.cpu_count()

genres = ["rock", "classical", "latin", "pop", "jazz", "soul",
          "classic bollywood", "rap", "folk", "funk", "opera"]
features = ['acousticness', 'instrumentalness', 'loudness', 'energy',
            'danceability', 'valence']
cluster_method = AgglomerativeClustering(n_clusters=len(genres))
# cluster_method = MiniBatchKMeans(n_clusters=len(genres))
model_prefix = '_feature_selection_without_explicit'
# model_prefix = '_feature_selection'


# Leave the rest
n_epochs = sum(epochs)
cluster_method_name = type(cluster_method).__name__

print('Reading data...')
tracks = pd.read_csv('data/tracks_with_genres.csv')
tracks['genres'] = tracks['genres'].apply(ast.literal_eval)
tracks = tracks.explode('genres')
tracks.rename(columns={'genres': 'genre'}, inplace=True)
tracks_subset = tracks[tracks['genre'].isin(genres)]

rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(tracks_subset, tracks_subset['genre'])

print('Transforming data...')
tracks_values = X[features].values
tracks_values = (tracks_values - tracks_values.mean(axis=0))\
                / tracks_values.std(axis=0)

rng = np.random.default_rng()
rng.shuffle(tracks_values, axis=0)

feature_combinations = []

for i in range(min_features, max_features+1):
    feature_combinations += list(itertools.combinations(features, i))

map_params = [{'index': i+1, 'features': features}
              for i, features in enumerate(feature_combinations)]

with open(f'output/{cluster_method_name}{model_prefix}_models.txt', 'w') as models_list_file:
    for params in map_params:
        features_subset = params['features']
        index = params['index']

        write_string = f'Model {index}: \n' \
                       f' Features: {", ".join(features_subset)}' \
                       f'\n\n'

        models_list_file.write(write_string)


def no_decay(param, t, max_iter):
    return param


def evaluate_model(params):
    index = params['index']
    features_subset = params['features']
    features_indices = [features.index(feature) for feature in features_subset]
    features_string = ', '.join(features_subset)

    learning_rate = np.concatenate(
        [np.linspace(*lr, epoch + 1)[:-1] for lr, epoch
         in zip(learning_rates, epochs)])
    sigma = np.concatenate(
        [np.linspace(*sig, epoch + 1)[:-1] for sig, epoch
         in zip(sigmas, epochs)])

    print(f'Running model {index} of {len(map_params)} with features: {features_string}')
    som = MiniSom(som_neurons[0], som_neurons[1], len(features_subset),
                  decay_function=no_decay)
    som.pca_weights_init(tracks_values[:, features_indices])  # For faster convergence?

    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for epoch in range(n_epochs):
        print(f'Model {index} running epoch {epoch+1}/{n_epochs}')
        som._sigma = sigma[epoch]
        som._learning_rate = learning_rate[epoch]
        som.train(tracks_values[:, features_indices], tracks_values.shape[0],
                  verbose=False, random_order=True)

        som_weights = som.get_weights()
        som_weights_flat = som_weights.reshape(som_neurons[0] * som_neurons[1],
                                               len(features_subset))
        labels = cluster_method.fit_predict(som_weights_flat)
        silhouette = silhouette_score(som_weights_flat, labels)
        davies_bouldin = davies_bouldin_score(som_weights_flat, labels)
        calinski_harabasz = calinski_harabasz_score(som_weights_flat, labels)

        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        calinski_harabasz_scores.append(calinski_harabasz)

        lines = []

        fig, ax1 = plt.subplots()
        x_range = range(1, len(silhouette_scores)+1)
        lines += ax1.plot(x_range, silhouette_scores, color='b',
                 linestyle='-', label='Silhouette')
        lines += ax1.plot(x_range, davies_bouldin_scores, color='b',
                 linestyle='-.', label='Davies-Bouldin')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_ylim(0, 2)
        ax1.set_xlabel('Epochs')

        ax2 = plt.twinx()
        lines += ax2.plot(x_range, calinski_harabasz_scores, color='r',
                 linestyle=':', label='Calinski-Harabasz')
        ax2.tick_params(axis='y', labelcolor='r')
        #ax2.set_ylim(0, 1000)

        labels = [line.get_label() for line in lines]

        ax1.legend(lines, labels)
        fig.tight_layout()

        cluster_method_name = type(cluster_method).__name__

        plt.savefig(f'output/figures/{cluster_method_name}{model_prefix}'
                    f'_model_{index}_training.png')
        plt.close()

        with open(f'models/{cluster_method_name}{model_prefix}_model_'
                  f'{index}.p', 'wb') as model_file:
            pickle.dump(som, model_file)

        print(f'Model {index} done with epoch {epoch+1}.')

    print(f'Model {index} finished.')
    return davies_bouldin_scores[-1]


print(f'Training {len(map_params)} models...')
pool = multiprocessing.Pool(min(multiprocessing.cpu_count(),
                                len(map_params)))
davies_bouldin_scores = pool.map(evaluate_model, map_params)
pool.close()
pool.join()

n_top_models = 5

print(f'\nTop {n_top_models} models:')
with open(f'output/{cluster_method_name}{model_prefix}_models_'
          f'scores.txt', 'w') as best_models_file:
    for count, index in enumerate(np.argsort(davies_bouldin_scores)):
        output = f'Model {index+1}, with a DB score = ' \
                 f'{davies_bouldin_scores[index]}'
        best_models_file.write(output + '\n')

        if count < n_top_models:
            print(output)
