import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d

# -------------------------------
# 1. Simulate the Signal (Repeat)
# -------------------------------

fs = 75
duration = 70
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
f1 = 1.55
f2 = 1.58
harmonic_range = np.arange(2, 9)
signal1 = np.zeros_like(t)
signal2 = np.zeros_like(t)
for n in harmonic_range:
    signal1 += 1/n * np.sin(2 * np.pi * n * f1 * t) * (1.0 + 0.01 * n)
    signal2 += 0.2 * n * np.sin(2 * np.pi * n * f2 * t) * (1.0 + 0.01 * n)
signal = signal1 + signal2
noise = 20 * np.random.randn(len(t))
signal = signal + noise

plt.plot(signal)

# -------------------------------
# 2. Compute the Average Spectrum
# -------------------------------

win_duration = 24  # seconds
win_length = int(fs * win_duration)
hop_length = int(win_length / 2)
f, time_stft, Zxx = stft(signal, fs, window='hann', nperseg=win_length,
                         noverlap=hop_length)

print(Zxx.shape)
avg_magnitude = np.mean(np.abs(Zxx), axis=1)

# Plot the average spectrum for reference.
plt.figure(figsize=(10, 4))
plt.plot(f, avg_magnitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Average Magnitude Spectrum")
#plt.xlim(0, 500)
plt.show()

# -------------------------------
# 3. Define and Compute the Pitch Saliency Function
# -------------------------------

# Set candidate grid for fundamentals (from 1 Hz to 5 Hz).
f_min_candidate = 1.0
f_max_candidate = 5.0
candidates = np.linspace(f_min_candidate, f_max_candidate, 400)  # 400 candidate values

# We consider harmonics n = 15 to 40.
n_min = 2
n_max = 15
saliency_scores = np.zeros_like(candidates)

# Create an interpolation function to sample the average spectrum smoothly.
interp_spectrum = interp1d(f, avg_magnitude, bounds_error=False, fill_value=0.0)

# For each candidate f0, sum the spectral magnitude at its harmonic frequencies.
for i, candidate in enumerate(candidates):
    salience = 0
    for n in range(n_min, n_max + 1):
        target = candidate * n
        if target > f[-1]:
            break
        salience += interp_spectrum(target)
    saliency_scores[i] = salience

plt.figure(figsize=(10, 4))
plt.plot(candidates, saliency_scores)
plt.xlabel("Candidate Fundamental (Hz)")
plt.ylabel("Saliency Score")
plt.title("Pitch Saliency Function")
plt.show()

# -------------------------------
# 4. Iterative Extraction of Two Fundamentals
# -------------------------------

# First extraction: candidate with maximum saliency.
idx_max = np.argmax(saliency_scores)
first_f0 = candidates[idx_max]
print("Method 2 – First extracted fundamental: {:.3f} Hz".format(first_f0))

# Subtract the first candidate's harmonic contributions.
residual_spectrum = avg_magnitude.copy()
for n in range(n_min, n_max + 1):
    target = first_f0 * n
    if target > f[-1]:
        break
    idx = np.argmin(np.abs(f - target))
    # Reduce the spectral magnitude (for example, set it to zero)
    residual_spectrum[idx] = 0

# Compute the residual pitch saliency.
interp_residual = interp1d(f, residual_spectrum, bounds_error=False, fill_value=0.0)
saliency_scores_res = np.zeros_like(candidates)
for i, candidate in enumerate(candidates):
    salience = 0
    for n in range(n_min, n_max + 1):
        target = candidate * n
        if target > f[-1]:
            break
        salience += interp_residual(target)
    saliency_scores_res[i] = salience

plt.figure(figsize=(10, 4))
plt.plot(candidates, saliency_scores_res)
plt.xlabel("Candidate Fundamental (Hz)")
plt.ylabel("Residual Saliency Score")
plt.title("Residual Pitch Saliency Function (After Subtraction)")
plt.show()

# Second extraction: candidate with maximum residual saliency.
idx_max_res = np.argmax(saliency_scores_res)
second_f0 = candidates[idx_max_res]
print("Method 2 – Second extracted fundamental: {:.3f} Hz".format(second_f0))
