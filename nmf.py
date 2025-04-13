import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# Set a random seed for reproducibility
np.random.seed(0)

# Define frequency and time axes for our synthetic spectrogram.
n_freq = 100      # number of frequency bins
n_time = 50       # number of time frames
freq = np.linspace(0, 100, n_freq)
time = np.linspace(0, 10, n_time)

# Define a helper function to generate a Gaussian profile.
def gaussian(f, center, width):
    return np.exp(-0.5 * ((f - center) / width) ** 2)

# Create two true basis functions (columns) that might represent the spectral signature
# of two different SONAR returns/harmonics.
W_true = np.zeros((n_freq, 2))
W_true[:, 0] = gaussian(freq, center=30, width=5)
W_true[:, 1] = gaussian(freq, center=60, width=8)

# Normalize the basis columns to have maximum value of 1.
W_true = W_true / np.max(W_true, axis=0)

# Define time-varying activations for the two components.
# These activations could correspond, for example, to how a SONAR echo varies over time.
H_true = np.zeros((2, n_time))
H_true[0, :] = np.sin(np.linspace(0, np.pi, n_time)) ** 2   # A squared sine pattern for component 1
H_true[1, :] = np.cos(np.linspace(0, np.pi, n_time)) ** 2   # A squared cosine pattern for component 2

# Generate the synthetic spectrogram-like data matrix: V = W_true * H_true.
V = np.dot(W_true, H_true)

# Add some small nonnegative noise (to simulate a real noisy measurement)
noise_level = 0.05
V_noise = V + noise_level * np.random.rand(n_freq, n_time)
V_noise = np.clip(V_noise, a_min=0, a_max=None)  # ensure nonnegativity

# Apply NMF to decompose the noisy spectrogram back into basis functions and activations.
nmf_model = NMF(n_components=2, init='random', random_state=0, max_iter=500)
W_est = nmf_model.fit_transform(V_noise)  # Estimated basis (frequency signatures)
H_est = nmf_model.components_             # Estimated time activations

# Plot and compare the true and estimated basis functions.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(freq, W_true[:, 0], label='True Basis 1', color='blue')
plt.plot(freq, W_true[:, 1], label='True Basis 2', color='red')
plt.title('True Basis Functions')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(freq, W_est[:, 0], label='Estimated Basis 1', color='blue')
plt.plot(freq, W_est[:, 1], label='Estimated Basis 2', color='red')
plt.title('Estimated Basis Functions from NMF')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

plt.tight_layout()
plt.show()

# Plot and compare the true and estimated activations.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, H_true[0, :], label='True Activation 1', color='blue', marker='o')
plt.plot(time, H_true[1, :], label='True Activation 2', color='red', marker='o')
plt.title('True Activations')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, H_est[0, :], label='Estimated Activation 1', color='blue', marker='o')
plt.plot(time, H_est[1, :], label='Estimated Activation 2', color='red', marker='o')
plt.title('Estimated Activations from NMF')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
