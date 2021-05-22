import ast
from collections import Counter

from imblearn.under_sampling import RandomUnderSampler
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

# Set the number of neurons to the same number as the loaded model
som_neurons = (38, 38)
model_id = 5
# model_prefix = '_without_explicit'
model_prefix = ''
ClusterClass = AgglomerativeClustering
# ClusterClass = MiniBatchKMeans
features = ['acousticness', 'instrumentalness', 'loudness', 'energy',
            'danceability', 'valence', 'explicit']

genres = ['rock', 'classical', 'latin', 'pop', 'jazz', 'soul',
          'classic bollywood', 'rap', 'folk', 'funk', 'opera']
clusters = len(genres)

rng = np.random.default_rng()
cluster_class_name = ClusterClass.__name__
model_name = f'{cluster_class_name}{model_prefix}_model_{model_id}'
model_path = f'models/{model_name}.p'

tracks = pd.read_csv('data/tracks_with_genres.csv')
tracks['genres'] = tracks['genres'].apply(ast.literal_eval)
tracks = tracks.explode('genres')
tracks.rename(columns={'genres': 'genre'}, inplace=True)
tracks_subset = tracks[tracks['genre'].isin(genres)]

rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(tracks_subset, tracks_subset['genre'])

tracks_genres = y.astype('category')

print(f'In total, there are {y.shape[0]} tracks in the dataset used.')

tracks_values = X[features].values
tracks_values = (tracks_values -
                 tracks_values.mean(axis=0))/tracks_values.std(axis=0)
rng.shuffle(tracks_values, axis=0)


def no_decay(param, t, max_iter):
    return param


# Load model
with open(model_path, 'rb') as model_file:
    som = pickle.load(model_file)


neuron_activations = som.activation_response(tracks_values)

sns.set_style('white')
ax = sns.heatmap(neuron_activations,
                 cbar_kws={'label': 'Number of activations'})
ax.get_figure().tight_layout()
plt.axis('off')
plt.show()
ax.get_figure().savefig(f'output/figures/{model_name}_activations.png',
                        bbox_inches='tight')
print(f'Activation map saved to output/figures/{model_name}_activations.png')


sns.set_style('white')
ax = sns.heatmap(som.distance_map(), cbar_kws={'label': 'Distance'})
ax.get_figure().tight_layout()
plt.axis('off')
plt.show()
ax.get_figure().savefig(f'output/figures/{model_name}_distance.png',
                        bbox_inches='tight')
print(f'Distance map saved to output/figures/{model_name}_distance.png')

cluster = ClusterClass(n_clusters=clusters)
som_weights = som.get_weights()
labels = cluster.fit_predict(
    som_weights.reshape(som_neurons[0]*som_neurons[1], len(features))
)

labels_matrix = labels.reshape(som_neurons[0], som_neurons[1])
ax = sns.heatmap(labels_matrix, cmap=sns.color_palette('Paired', clusters))
fig = ax.get_figure().tight_layout()

cbar = ax.collections[0].colorbar
tick_locs = np.linspace(0.5, clusters-1.5, clusters)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(list(range(clusters)))

ax.axis('off')
ax.get_figure().savefig(f'output/figures/{model_name}_clusters.png',
                        bbox_inches='tight')
plt.show()
print(f'Cluster map saved to output/figures/{model_name}_clusters.png')

labels_map = som.labels_map(tracks_values, tracks_genres)
labels_classified_as_genre = [Counter() for i in range(clusters)]

for neuron in range(len(labels)):
    y, x = neuron//som_neurons[1], neuron % som_neurons[1]
    label = labels[neuron]
    counter = labels_map[(y, x)]

    if counter:
        labels_classified_as_genre[label] += counter

classifications = pd.DataFrame(labels_classified_as_genre)
_, ax = plt.subplots()

colormap = ListedColormap(sns.color_palette('Paired', clusters).as_hex())
fig = classifications.T.plot(kind='bar', figsize=(20, 16), fontsize=12,
                             stacked=True, ax=ax, cmap=colormap).get_figure()
ax.set_ylabel('Number of songs', fontsize='40')
ax.set_xlabel('Genre', fontsize='40')
handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles), reversed(labels), title='Cluster',
          title_fontsize='40', bbox_to_anchor=(1.0, 0.8),
          mode='none', ncol=1, fontsize='35', framealpha=0.6)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(30)
fig.tight_layout()
plt.savefig(f'output/figures/{model_name}_genre_clusters_bars.png')
plt.show()
print(f'Genre distribution plot saved to output/figures/{model_name}'
      f'_genre_clusters_bars.png')
