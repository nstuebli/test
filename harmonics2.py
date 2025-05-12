import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate

def autocorrelate(x):
    """
    Compute the autocorrelation of signal x.
    We use 'full' mode to get correlation at negative and positive lags,
    then take only the second half (lags ≥ 0).
    """
    corr_full = correlate(x, x, mode='full', method='fft')
    mid = corr_full.size // 2
    return corr_full[mid:]

def find_fundamental_via_ac(signal, fs, min_lag=20, peak_threshold=0.75):
    """
    Estimate the fundamental frequency (f0) of a signal using its autocorrelation.
    
    Parameters:
      signal : np.array
          Input time-domain signal.
      fs : float
          Sample rate in Hz.
      min_lag : int, optional
          Minimum lag to consider in autocorrelation (to avoid the zero-lag peak and very short lags).
      peak_threshold : float, optional
          Minimum normalized height for a peak to be considered [0, 1].
    
    Returns:
      f0 : float or None
          The estimated fundamental frequency in Hz, or None if no suitable peak is found.
      ac : np.array
          The computed autocorrelation of the signal.
      peak_lags : np.array
          Indices (lags) of the detected peaks in the autocorrelation function.
    """
    # Remove DC offset
    signal = signal - np.mean(signal)
    
    # Compute autocorrelation and normalize so the zero-lag value is 1
    ac = autocorrelate(signal)
    ac = ac / ac[0]

    # Find peaks in the autocorrelation; ignore the zero lag by specifying a minimum lag
    # Here, we start looking from index 'min_lag'
    peaks, properties = find_peaks(ac[min_lag:], height=peak_threshold)
    
    # Adjust the peak indices because we started at min_lag
    peak_lags = peaks + min_lag

    if len(peak_lags) == 0:
        print("No significant peak found!")
        return None, ac, None

    # The first peak after lag zero is taken as the period of the signal
    fundamental_lag = peak_lags[0]
    f0 = fs / fundamental_lag
    return f0, ac, peak_lags

def pitch_from_cepstrum(signal, fs, min_quefrency_sec=0.002, max_quefrency_sec=0.02):
    """
    Estimate the fundamental frequency using the real cepstrum.
    
    Parameters:
      signal : np.array
          Input time-domain signal.
      fs : float
          Sampling rate in Hz.
      min_quefrency_sec : float
          Minimum quefrency (in seconds) to consider. This sets the lower bound for the expected pitch period.
      max_quefrency_sec : float
          Maximum quefrency (in seconds) to consider. This sets the upper bound for the expected pitch period.
    
    Returns:
      fundamental_freq : float
          The estimated fundamental frequency in Hz.
      cepstrum : np.array
          The computed real cepstrum.
      quefrency_axis : np.array
          Quefrency axis (in seconds).
    """
    # Remove DC offset
    signal = signal - np.mean(signal)
    
    # Apply a Hann window to your signal:
    window = np.hanning(len(signal))
    windowed_signal = signal * window
    
    # Compute the spectrum of the signal
    spectrum = np.fft.fft(signal)
    
    # Obtain the log magnitude spectrum. We add a small epsilon to prevent log(0).
    log_magnitude = np.log(np.abs(spectrum) + np.finfo(float).eps)
    
    # Compute the real cepstrum using the inverse FFT of the log magnitude spectrum
    cepstrum = np.fft.ifft(log_magnitude).real
    
    # Create quefrency axis in seconds
    n = len(signal)
    quefrency_axis = np.arange(n) / fs
    
    # Define search region for the pitch period (in samples)
    min_quefrency = int(np.floor(min_quefrency_sec * fs))
    max_quefrency = int(np.ceil(max_quefrency_sec * fs))
    
    # Find the peak within the specified quefrency range
    peak_region = cepstrum[min_quefrency:max_quefrency]
    peaks, _ = find_peaks(peak_region)
    
    if len(peaks) == 0:
        print("No clear peak found in the cepstrum within the expected pitch range!")
        return None, cepstrum, quefrency_axis
    
    # Adjust peak indices relative to the whole cepstrum
    peak_idx = peaks[np.argmax(peak_region[peaks])] + min_quefrency
    pitch_period = quefrency_axis[peak_idx]
    fundamental_freq = 1.0 / pitch_period
    
    return fundamental_freq, cepstrum, quefrency_axis

def short_time_cepstrum(signal, fs, segment_length, overlap, window_type='hann', lifter_index=0):
    step = segment_length - overlap
    segments = []
    cepstra = []
    for start in range(0, len(signal) - segment_length + 1, step):
        segment = signal[start:start + segment_length]
        window = get_window(window_type, segment_length)
        segment = segment * window
        cep = compute_cepstrum(segment, fs)
        # Optionally apply liftering to emphasize pitch-related peaks
        if lifter_index > 0:
            cep = lifter_cepstrum(cep, lifter_index)
        cepstra.append(cep)
        segments.append(segment)
    
    # For demonstration, average the cepstra:
    avg_cepstrum = np.mean(cepstra, axis=0)
    return avg_cepstrum, cepstra

if __name__ == "__main__":
    # Parameters
    fs = 2000          # Sampling frequency in Hz
    duration = 20.0      # Signal duration in seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # True (missing) fundamental frequency
    true_f0 = 3.57  # Hz
    
    # Create a signal using only harmonics of the fundamental (e.g., 2nd, 3rd, and 4th harmonics)
    harmonics = [2, 3, 7, 8, 11, 13, 21, 22, 24, 30, 40, 52, 77]
    signal = np.zeros_like(t)
    for h in harmonics:
        signal += np.sin(2 * np.pi * true_f0 * h * t)
    """
    # Estimate the fundamental frequency via autocorrelation
    estimated_f0, ac, peak_lags = find_fundamental_via_ac(signal, fs, min_lag=fs//250, peak_threshold=0.5)
    print("Estimated fundamental frequency: {:.2f} Hz".format(estimated_f0))
    
    # Plot the autocorrelation function and mark detected peaks
    plt.figure(figsize=(10, 4))
    plt.plot(ac, label="Autocorrelation")
    if peak_lags is not None:
        plt.plot(peak_lags, ac[peak_lags], 'ro', label="Detected peaks")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Normalized autocorrelation")
    plt.legend()
    plt.title("Autocorrelation Function and Detected Peaks")
    plt.xlim(0,100)
    plt.show()
    """
    
    # Estimate the fundamental frequency using cepstral analysis
    estimated_f0, cepstrum, quefrency_axis = pitch_from_cepstrum(signal, fs, 0.5, 1.0)
    print(f"Estimated fundamental frequency: {estimated_f0:.2f} Hz")
    
    # Plot the cepstrum to visualize the peak in the expected range
    plt.figure(figsize=(10, 4))
    plt.plot(quefrency_axis, cepstrum, label="Cepstrum")
    plt.xlabel("Quefrency (s)")
    plt.ylabel("Amplitude")
    plt.title("Cepstrum of the Signal")
    plt.axvline(x=1.0/estimated_f0, color='r', linestyle='--', label="Estimated Pitch Period")
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(-0.05, 0.1)
    plt.show()
