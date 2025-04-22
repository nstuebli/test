namespace Helper
{
    struct WavHeader
    {
        char riff[4];            // "RIFF"
        uint32_t fileSize;       // File size minus 8 bytes
        char wave[4];            // "WAVE"
        char fmt[4];             // "fmt "
        uint32_t fmtSize;        // Size of the fmt chunk
        uint16_t format;         // Format type (1 for PCM)
        uint16_t channels;       // Number of channels
        uint32_t sampleRate;     // Sampling rate
        uint32_t byteRate;       // Byte rate
        uint16_t blockAlign;     // Block align
        uint16_t bitsPerSample;  // Bits per sample
    };

    std::vector<int16_t> readWavFile(const std::string& filePath)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open file");
        }

        WavHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        // Verify the WAV file format
        if (std::string(header.riff, 4) != "RIFF" ||
            std::string(header.wave, 4) != "WAVE" ||
            std::string(header.fmt, 4) != "fmt ") {
            throw std::runtime_error("Invalid WAV file");
        }

        // Verify PCM format and 16-bit samples
        if (header.format != 1 || header.bitsPerSample != 16)
        {
            throw std::runtime_error("Unsupported WAV file format");
        }

        // Skip any extra fmt bytes
        if (header.fmtSize > 16)
        {
            file.seekg(header.fmtSize - 16, std::ios::cur);
        }

        // Find the "data" chunk
        char chunkId[4];
        uint32_t dataSize;
        while (file.read(chunkId, 4))
        {
            file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            if (std::string(chunkId, 4) == "data")
            {
                break;
            }
            // Skip over the chunk if it's not "data"
            file.seekg(dataSize, std::ios::cur);
        }

        // Read audio data
        std::vector<int16_t> audioData(dataSize / sizeof(int16_t));
        file.read(reinterpret_cast<char*>(audioData.data()), dataSize);

        return audioData;
    }

    std::vector<float> readWavFile(const std::string& filePath)
    {
        std::ifstream file(filePath, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open file");
        }

        WavHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        // Verify the WAV file format
        if (std::string(header.riff, 4) != "RIFF" ||
            std::string(header.wave, 4) != "WAVE" ||
            std::string(header.fmt, 4) != "fmt ")
        {
            throw std::runtime_error("Invalid WAV file");
        }

        // Verify IEEE Float format (format 3) and 32-bit samples
        if (header.format != 3 || header.bitsPerSample != 32)
        {
            throw std::runtime_error("Unsupported WAV file format");
        }

        // Skip any extra fmt bytes
        if (header.fmtSize > 16)
        {
            file.seekg(header.fmtSize - 16, std::ios::cur);
        }

        // Find the "data" chunk
        char chunkId[4];
        uint32_t dataSize;
        while (file.read(chunkId, 4))
        {
            file.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            if (std::string(chunkId, 4) == "data")
            {
                break;
            }
            // Skip over the chunk if it's not "data"
            file.seekg(dataSize, std::ios::cur);
        }

        // Read audio data (32-bit floating point samples)
        std::vector<float> audioData(dataSize / sizeof(float));
        file.read(reinterpret_cast<char*>(audioData.data()), dataSize);

        return audioData;
    }
}
