import numpy as np
import matplotlib.pyplot as plt

def music_spectrum(x, M, d, n_freqs=1024):
    """
    Computes the MUSIC pseudospectrum.
    
    Parameters:
      x       : 1D complex signal (length N)
      M       : Subarray (or snapshot) size, an integer smaller than N
      d       : Number of signal sources (e.g., 2 for two sinusoids)
      n_freqs : Number of points in the frequency grid
      
    Returns:
      f_grid  : Frequency grid (normalized frequency)
      P_db    : MUSIC pseudospectrum in dB
    """
    N = len(x)
    L = N - M + 1  # Number of subarrays (snapshots)

    # Form the data matrix using overlapping segments (Hankel matrix)
    X = np.zeros((M, L), dtype=complex)
    for i in range(L):
        X[:, i] = x[i:i+M]

    # Estimate the covariance matrix
    R = np.dot(X, X.conj().T) / L

    # Eigen-decomposition of covariance matrix
    eigvals, eigvecs = np.linalg.eig(R)
    # Sort eigenvalues in ascending order; the smallest belong to the noise subspace
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Noise subspace: columns corresponding to the M - d smallest eigenvalues
    En = eigvecs[:, :M - d]

    # Create a frequency grid to evaluate the pseudospectrum
    f_grid = np.linspace(0, 1, n_freqs, endpoint=False)  # normalized frequency grid [0,1)
    P = np.zeros(n_freqs)
    for j, f in enumerate(f_grid):
        # Steering vector: a(f) = [1, exp(-j2πf), ..., exp(-j2πf*(M-1))]^T
        a = np.exp(-1j * 2 * np.pi * f * np.arange(M))
        # MUSIC spectrum denominator: a^H * En * En^H * a; use norm squared.
        denom = np.linalg.norm( np.dot(En.conj().T, a) )**2
        # Avoid division by zero issues.
        P[j] = 1 / (denom + 1e-12)

    # Convert to decibels for clearer visualization.
    P_db = 10 * np.log10(P)
    return f_grid, P_db

# ----------------- Synthetic Data Simulation ----------------- #

# Number of samples in the signal
N = 200
n = np.arange(N)

# Two closely spaced normalized frequencies (e.g., 0.15 and 0.16 of the sampling rate)
f1 = 0.15  
f2 = 0.16  
A1 = 1.0  # amplitude for the first sinusoid
A2 = 0.8  # amplitude for the second sinusoid

# Generate a complex signal as a sum of two complex exponentials
x = A1 * np.exp(1j * 2 * np.pi * f1 * n) + A2 * np.exp(1j * 2 * np.pi * f2 * n)

# Add complex Gaussian noise
noise_power = 0.1
noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
x = x + noise

# ----------------- MUSIC Parameters ----------------- #

M = 50  # Subarray length (choose M < N; larger M can improve resolution but requires more data)
d = 2   # We expect 2 sources (i.e., 2 sinusoids)

# Compute the MUSIC pseudospectrum
f_grid, P_db = music_spectrum(x, M, d)

# ----------------- Plotting the Result ----------------- #

plt.figure(figsize=(10, 6))
plt.plot(f_grid, P_db, label='MUSIC Pseudospectrum')
plt.xlabel("Normalized Frequency")
plt.ylabel("Spectrum (dB)")
plt.title("MUSIC Superresolution Spectrum")
plt.grid(True)
# Mark estimated peaks: In a more elaborate version, you could detect the peaks in 'P_db'
plt.axvline(f1, color='red', linestyle='--', label=f"True f1 = {f1}")
plt.axvline(f2, color='green', linestyle='--', label=f"True f2 = {f2}")
plt.legend()
plt.show()
