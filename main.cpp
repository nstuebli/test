#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>

// -----------------------------------------------------------------
// 1. Linear Interpolation Helper Function
// -----------------------------------------------------------------
// Given a sorted frequency vector `xs` and corresponding magnitude vector `ys`,
// this function returns the linearly interpolated value at x_val.
float linearInterpolate(const std::vector<float>& xs, 
                        const std::vector<float>& ys, 
                        float x_val) {
    if (x_val <= xs.front())
        return ys.front();
    if (x_val >= xs.back())
        return ys.back();

    auto it = std::upper_bound(xs.begin(), xs.end(), x_val);
    size_t idx = std::distance(xs.begin(), it);
    size_t i1 = (idx == 0) ? 0 : idx - 1;
    size_t i2 = idx;
    float x1 = xs[i1], x2 = xs[i2];
    float y1 = ys[i1], y2 = ys[i2];
    return y1 + (y2 - y1) * ((x_val - x1) / (x2 - x1));
}

// -----------------------------------------------------------------
// 2. Compute Pitch Saliency Function (Step 3)
// -----------------------------------------------------------------
// Given a frequency vector, an averaged spectrum, a candidate grid and a range of harmonics,
// returns a vector containing the saliency score for each candidate fundamental.
std::vector<float> computePitchSaliency(
    const std::vector<float>& freq,
    const std::vector<float>& avgMagnitude,
    const std::vector<float>& candidates,
    int n_min,
    int n_max)
{
    std::vector<float> saliency_scores(candidates.size(), 0.0f);
    for (size_t i = 0; i < candidates.size(); i++) {
        float candidate = candidates[i];
        float salience = 0.0f;
        for (int n = n_min; n <= n_max; ++n) {
            float target = candidate * n;
            if (target > freq.back())
                break;
            salience += linearInterpolate(freq, avgMagnitude, target);
        }
        saliency_scores[i] = salience;
    }
    return saliency_scores;
}

// -----------------------------------------------------------------
// 3. Iterative Extraction of Two Fundamentals (Step 4)
// -----------------------------------------------------------------
// This routine performs two iterations:
//  1. It computes the saliency on the original spectrum and selects the candidate with 
//     maximum saliency as the first fundamental.
//  2. It subtracts (zeroes out) the first fundamental’s harmonic contributions from a copy 
//     of the averaged spectrum, recomputes the saliency on the residual, and selects the second.
std::pair<float, float> extractFundamentals(
    const std::vector<float>& freq,
    const std::vector<float>& avgMagnitude,
    const std::vector<float>& candidates,
    int n_min,
    int n_max)
{
    // Compute saliency of original spectrum.
    std::vector<float> saliency_scores = computePitchSaliency(freq, avgMagnitude, candidates, n_min, n_max);
    
    // First extraction: candidate with maximum saliency.
    auto max_it = std::max_element(saliency_scores.begin(), saliency_scores.end());
    size_t idx_max = std::distance(saliency_scores.begin(), max_it);
    float first_f0 = candidates[idx_max];
    std::cout << "First extracted fundamental: " << first_f0 << " Hz" << std::endl;
    
    // Create a residual spectrum by copying the averaged spectrum.
    std::vector<float> residualSpectrum = avgMagnitude;
    // Subtract the first candidate's harmonic contributions:
    // For harmonic orders n in [n_min, n_max], find the closest frequency bin to candidate*f0
    // and set that value to zero.
    for (int n = n_min; n <= n_max; n++) {
        float target = first_f0 * n;
        if (target > freq.back())
            break;
        auto it = std::min_element(freq.begin(), freq.end(), [target](float a, float b) {
            return std::fabs(a - target) < std::fabs(b - target);
        });
        int idx = std::distance(freq.begin(), it);
        residualSpectrum[idx] = 0.0f;  // Zero out this harmonic bin.
    }
    
    // Compute the saliency on the residual spectrum.
    std::vector<float> saliency_scores_res = computePitchSaliency(freq, residualSpectrum, candidates, n_min, n_max);
    auto max_res_it = std::max_element(saliency_scores_res.begin(), saliency_scores_res.end());
    size_t idx_max_res = std::distance(saliency_scores_res.begin(), max_res_it);
    float second_f0 = candidates[idx_max_res];
    std::cout << "Second extracted fundamental: " << second_f0 << " Hz" << std::endl;
    
    return std::make_pair(first_f0, second_f0);
}

// -----------------------------------------------------------------
// Example main() demonstrating Steps 3 and 4
// -----------------------------------------------------------------
int main() {
    // -----------------------------------------------------------------
    // Pre-assumed inputs:
    // You already have the averaged spectrum 'avgMagnitude' and frequency axis 'freq'.
    // In a real application, these will be computed from your spectrogram of the input signal.
    //
    // For this example, we'll simulate simple vectors.
    // Assume freq goes from 0 to 500 Hz in 0.5 Hz increments.
    // And avgMagnitude is a vector of the same length representing the spectrum.
    // -----------------------------------------------------------------
    
    std::vector<float> freq;
    for (float f_val = 0.0f; f_val <= 500.0f; f_val += 0.5f)
        freq.push_back(f_val);
    
    // Here we create a dummy averaged spectrum.
    // In your case, replace this with your actual computed avgMagnitude.
    std::vector<float> avgMagnitude(freq.size(), 0.0f);
    // For demonstration, let's simulate a scenario where two harmonic series exist.
    // Suppose the two fundamentals are approximately 2.57 Hz and 2.59 Hz,
    // but the energy is mostly in the harmonics (for orders 15 to 40).
    // We'll add some energy at the harmonic locations.
    int n_min = 15;
    int n_max = 40;
    float trueF1 = 2.57f;
    float trueF2 = 2.59f;
    for (int n = n_min; n <= n_max; n++) {
        float harmonic1 = n * trueF1;
        float harmonic2 = n * trueF2;
        if (harmonic1 <= freq.back()) {
            auto it = std::min_element(freq.begin(), freq.end(), [harmonic1](float a, float b) {
                return std::fabs(a - harmonic1) < std::fabs(b - harmonic1);
            });
            int idx = std::distance(freq.begin(), it);
            // For simplicity, add an arbitrary amplitude that grows with n.
            avgMagnitude[idx] += 1.0f + 0.01f * n;
        }
        if (harmonic2 <= freq.back()) {
            auto it = std::min_element(freq.begin(), freq.end(), [harmonic2](float a, float b) {
                return std::fabs(a - harmonic2) < std::fabs(b - harmonic2);
            });
            int idx = std::distance(freq.begin(), it);
            avgMagnitude[idx] += 1.0f + 0.01f * n;
        }
    }
    
    // -----------------------------------------------------------------
    // Set up candidate fundamentals.
    // We'll create a candidate grid from 1 Hz to 5 Hz with 400 points
    // (similar to np.linspace(1.0, 5.0, 400) in Python).
    // -----------------------------------------------------------------
    float f_min_candidate = 1.0f;
    float f_max_candidate = 5.0f;
    int num_candidates = 400;
    std::vector<float> candidates(num_candidates);
    float step = (f_max_candidate - f_min_candidate) / (num_candidates - 1);
    for (int i = 0; i < num_candidates; ++i) {
        candidates[i] = f_min_candidate + i * step;
    }
    
    // -----------------------------------------------------------------
    // Run the iterative extraction of two fundamentals:
    // Compute the pitch saliency function and subtract the first candidate's harmonics,
    // then compute the residual saliency to extract the second fundamental.
    // -----------------------------------------------------------------
    std::pair<float, float> fundamentals = extractFundamentals(freq, avgMagnitude, candidates, n_min, n_max);
    
    // Output the extracted fundamentals.
    std::cout << "\nFinal extracted fundamentals: " 
              << fundamentals.first << " Hz and " << fundamentals.second << " Hz" << std::endl;
              
    return 0;
}
