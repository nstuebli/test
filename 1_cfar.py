import numpy as np
import matplotlib.pyplot as plt

def parabolic_interpolation(signal, peak_index):
    """
    Refine the peak location using parabolic interpolation.

    Given a candidate peak at index `peak_index` (with neighbors on each side),
    estimate a refined position and amplitude by fitting a parabola.
    """
    if peak_index <= 0 or peak_index >= len(signal) - 1:
        return peak_index, signal[peak_index]
    
    # Values at the left, center, right of the peak
    y_m1 = signal[peak_index - 1]
    y_0  = signal[peak_index]
    y_p1 = signal[peak_index + 1]
    
    # Denominator for parabolic formula, check for division by zero.
    denominator = 2 * (y_m1 - 2 * y_0 + y_p1)
    if denominator == 0:
        offset = 0
    else:
        offset = (y_m1 - y_p1) / denominator
    refined_index = peak_index + offset
    refined_value = y_0 - 0.25 * (y_m1 - y_p1) * offset
    return refined_index, refined_value

def refined_thresholding(signal, window_size=20, guard_cells=3, threshold_factor=1.5):
    """
    Compute an adaptive (refined) threshold for each index
    using a sliding window of `window_size` around the target sample,
    while excluding `guard_cells` to avoid influence from a nearby peak.
    """
    thresholds = np.zeros_like(signal)
    N = len(signal)
    
    for i in range(N):
        # Consider a symmetric window around index i.
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)
        
        # Exclude the guard cells around index i.
        # Left noise cells: from start to (i - guard_cells)
        left_noise = np.arange(start, max(i - guard_cells, start))
        # Right noise cells: from (i + guard_cells + 1) to end
        right_noise = np.arange(min(i + guard_cells + 1, end), end)
        
        noise_indices = np.concatenate((left_noise, right_noise))
        
        if noise_indices.size == 0:
            # No noise data, mark threshold as NaN.
            thresholds[i] = np.nan
        else:
            # A robust noise estimate can be taken as the median of the local noise
            noise_estimate = np.median(signal[noise_indices])
            thresholds[i] = noise_estimate * threshold_factor
    return thresholds

def robust_peak_detection(signal, window_size=20, guard_cells=3, threshold_factor=1.5, min_distance=5):
    """
    Perform robust peak detection using refined thresholding and parabolic interpolation.
    
    Parameters:
      signal           : 1D numpy array containing the signal (e.g., a spectrum)
      window_size      : Number of samples on each side to consider for the local noise estimate.
      guard_cells      : Number of samples to exclude immediately around the target sample.
      threshold_factor : Scaling factor to determine the detection threshold.
      min_distance     : Minimum distance (in sample indices) between peaks.
    
    Returns:
      final_peaks      : List of tuples (refined_index, refined_value) for each detected peak.
      thresholds       : Array containing the local threshold at each point.
      candidate_peaks  : List of candidate peak indices before refinement.
    """
    thresholds = refined_thresholding(signal, window_size, guard_cells, threshold_factor)
    
    candidate_peaks = []
    # A candidate peak should be above its local threshold and a local maximum.
    for i in range(1, len(signal) - 1):
        if signal[i] > thresholds[i] and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            candidate_peaks.append(i)
    
    # Refine candidate peaks with parabolic interpolation.
    refined_peaks = []
    for peak in candidate_peaks:
        refined_index, refined_value = parabolic_interpolation(signal, peak)
        refined_peaks.append((refined_index, refined_value))
    
    # Enforce a minimum distance between peaks.
    # This simple method keeps the first peak and only adds a new one if far enough.
    refined_peaks.sort(key=lambda x: x[0])
    final_peaks = []
    if refined_peaks:
        final_peaks.append(refined_peaks[0])
        for rpeak in refined_peaks[1:]:
            if rpeak[0] - final_peaks[-1][0] >= min_distance:
                final_peaks.append(rpeak)
                
    return final_peaks, thresholds, candidate_peaks

# ----------------- Testing with a synthetic signal ----------------- #

if __name__ == '__main__':
    # Create a synthetic signal with two overlapping Gaussian peaks plus noise.
    x = np.linspace(0, 100, 1000)
    # Two Gaussian peaks with centers close to each other, e.g., at 48 and 48.5.
    peak1 = np.exp(-0.5 * ((x - 48) / 1.0) ** 2)
    peak2 = 0.8 * np.exp(-0.5 * ((x - 48.5) / 1.0) ** 2)
    noise = 0.05 * np.random.randn(len(x))
    signal = peak1 + peak2 + noise

    # Detect peaks using our robust method.
    peaks, thresholds, candidate_peaks = robust_peak_detection(
        signal,
        window_size=20,
        guard_cells=3,
        threshold_factor=1.5,
        min_distance=5
    )

    # Output the refined peak locations and amplitudes.
    print("Detected peaks (refined):")
    for pos, amp in peaks:
        print(f"Peak at index ≈ {pos:.2f}, amplitude: {amp:.2f}")

    # Plot the signal, threshold, and detected peaks.
    plt.figure(figsize=(12, 6))
    plt.plot(x, signal, label="Signal", color='blue')
    plt.plot(x, thresholds, label="Local Threshold", linestyle='--', color='red')
    for pos, amp in peaks:
        # Map back the refined peak position to the x-axis.
        plt.plot(x[int(round(pos))], amp, 'kx', markersize=10)
    plt.xlabel("X-axis (e.g., Frequency or Time)")
    plt.ylabel("Amplitude")
    plt.title("Robust Peak Detection with Refined Thresholding and Parabolic Interpolation")
    plt.legend()
    plt.show()
