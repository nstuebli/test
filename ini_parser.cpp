#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

std::unordered_map<std::string, std::string> LoadINI(const std::string& filename) {
        std::unordered_map<std::string, std::string> config;
        std::ifstream file(filename);
        std::string line, currentSection;

        if (!file) {
            std::cerr << "Error loading INI file.\n";
            return config;
        }

        while (std::getline(file, line)) {
            line = line.substr(0, line.find(';')); // Remove comments
            line.erase(0, line.find_first_not_of(" \t")); // Trim left
            line.erase(line.find_last_not_of(" \t") + 1); // Trim right

            if (line.empty()) continue;

            if (line.front() == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.size() - 2);
            } else {
                std::istringstream iss(line);
                std::string key, value;
                if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);

                    if (!currentSection.empty()) key = currentSection + "." + key;
                    config[key] = value;
                }
            }
        }
        return config;
    }

    WindowType ParseWindowType(const std::string& value) {
        if (value == "HANN") return WindowType::HANN;
        if (value == "HAMMING") return WindowType::HAMMING;
        if (value == "KAISER") return WindowType::KAISER;
        return WindowType::HANN; // Default fallback
    }

    void LoadParameters(const std::string& filename) {
        auto config = LoadINI(filename);

        THRESHOLD_STD_FACTOR = std::stof(config["General.THRESHOLD_STD_FACTOR"]);
        MASK_DECREASE = std::stof(config["General.MASK_DECREASE"]);
        DECIMATION_CUTOFF = std::stof(config["General.DECIMATION_CUTOFF"]);
        FLOWNOISE_CUTOFF = std::stof(config["General.FLOWNOISE_CUTOFF"]);
        FFT_SIZE = std::stoi(config["General.FFT_SIZE"]);
        HARMONICERROR_DELTA = std::stof(config["General.HARMONICERROR_DELTA"]);
        FFT_WINDOW = ParseWindowType(config["General.FFT_WINDOW"]);
    }
