#include <iostream>
#include <vector>

using namespace std;

// Function to perform CFAR detection on a given data vector.
// Parameters:
//   data         - input data vector (e.g., radar returns)
//   trainingCells- number of training cells on each side of the CUT
//   guardCells   - number of guard cells on each side of the CUT (cells ignored)
//   rateFactor   - scaling factor to determine the threshold from noise average
// Returns: Boolean detection vector with 'true' if a detection is made at that cell.
vector<bool> cfar(const vector<double>& data, int trainingCells, int guardCells, double rateFactor) {
    int N = data.size();
    vector<bool> detection(N, false);
    
    // CFAR processing can only be applied to cells that have enough data on both sides.
    // We start from index = trainingCells + guardCells, and go until index = N - (trainingCells + guardCells) - 1.
    for (int i = trainingCells + guardCells; i < N - trainingCells - guardCells; ++i) {
        double sumNoise = 0.0;
        
        // Sum noise from left-side training cells.
        for (int j = i - trainingCells - guardCells; j < i - guardCells; ++j) {
            sumNoise += data[j];
        }
        
        // Sum noise from right-side training cells.
        for (int j = i + guardCells + 1; j <= i + trainingCells + guardCells; ++j) {
            sumNoise += data[j];
        }
        
        // Total number of training cells is twice the trainingCells parameter.
        int numTrainingCells = trainingCells * 2;
        double noiseAvg = sumNoise / numTrainingCells;
        
        // The threshold is computed as a scaled noise average.
        double threshold = noiseAvg * rateFactor;
        
        // If the CUT exceeds the threshold, flag it as a potential target.
        if (data[i] > threshold) {
            detection[i] = true;
        }
    }
    
    return detection;
}

int main() {
    // An example input signal:
    // The background noise level is around 1. A strong pulse (value 50) is embedded as a target.
    vector<double> data = {
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };

    // CFAR parameters:
    int trainingCells = 3;   // Number of training cells on each side
    int guardCells = 1;      // Number of guard cells on each side
    double rateFactor = 2.5; // Threshold multiplier
    
    // Run CFAR detection on the data.
    vector<bool> detections = cfar(data, trainingCells, guardCells, rateFactor);
    
    // Output the detection results.
    cout << "CFAR Detection Results:" << endl;
    for (size_t i = 0; i < data.size(); i++) {
        cout << "Cell " << i << " (value: " << data[i] << "): " 
             << (detections[i] ? "Target detected" : "No target") << endl;
    }
    
    return 0;
}
