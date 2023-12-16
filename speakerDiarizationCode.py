from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import soundfile as sf
from sklearn import cluster
import numpy as np
from scipy.spatial import distance

# Reading the sound file and extracting features
(rate, sig) = wav.read("LDC2007S10.wav")
data = mfcc(sig, rate)
data = data.reshape(data.shape[0], 13)

# Using soundfile to read the audio file
dataset, fs = sf.read('LDC2007S10.wav')
dataset = dataset.reshape(dataset.shape[0], 1)

# Applying KMeans clustering
ks = range(1, 2)
KMeans = [cluster.KMeans(n_clusters=i, init="k-means++").fit(data[179989:179999]) for i in ks]

# Function to compute Bayesian Information Criterion (BIC)
def compute_bic(kmeans, X):
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    m = kmeans.n_clusters
    n = np.bincount(labels)
    N, d = X.shape
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * np.log(N) * (d+1)
    BIC =  np.sum([n[i] * np.log(n[i]) - n[i] * np.log(N) - ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) - ((n[i] - 1) * d/2) for i in range(m)]) - const_term
    return BIC

# Detecting speaker changes
threshold = 300
count = 0

for i in range(200, len(data)):
    X1 = data[i-200:i]
    X2 = data[i:i+200]
    X = data[i-200:i+200]

    if X2.shape[0] < 200:
        break

    bic1 = [compute_bic(kmeansi, X1) for kmeansi in KMeans]
    bic2 = [compute_bic(kmeansi, X2) for kmeansi in KMeans]
    bic = [compute_bic(kmeansi, X) for kmeansi in KMeans]
    diff = abs((bic1[0] + bic2[0]) - bic[0])

    if diff > threshold:
        print("Speaker change detected at frame", i)
        count += 1

print("Total speaker changes detected:", count)
