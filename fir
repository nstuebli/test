stop = std::min(stop, static_cast<int>(input.size()));
    int inputSize = input.size();
    int kernelSize = m_FIRCoefficiants.size();

    for (int i = start; i < stop; ++i) // Oder über input i += m_DownSampling?
    {
        __m256 _sum = _mm256_setzero_ps();
        int j = 0;
        // Process main part with AVX
        for (; j + 8 <= kernelSize; j += 8)
        {
            if (i * m_DownSampling - j >= 0 && i * m_DownSampling - j + 8 <= inputSize)
            {
                __m256 _inputVals = _mm256_loadu_ps(&input[i * m_DownSampling - j]); // i - j schlecht?
                __m256 _kernelVals = _mm256_loadu_ps(&m_FIRCoefficiants[j]);
                _sum = _mm256_fmadd_ps(_inputVals, _kernelVals, _sum);
            }
        }

        // Process remaining samples with normal loop
        for (; j < kernelSize; ++j)
        {
            if (i * m_DownSampling - j >= 0 && i * m_DownSampling - j < inputSize)
            {
                output[i] += input[i * m_DownSampling - j] * m_FIRCoefficiants[j];
            }
        }

        float temp[8];
        _mm256_storeu_ps(temp, _sum);
        output[i] += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }


// reverse-lane index: lane-0←7, lane-1←6, …, lane-7←0
const __m256i REV_IDX = _mm256_setr_epi32(7,6,5,4,3,2,1,0);

for (int i = start; i < stop; ++i) 
{
    // i indexes your output[], which is input.size()/m_DownSampling big
    // zero it before accumulation:
    output[i] = 0.0f;

    __m256 _sum = _mm256_setzero_ps();
    int j = 0;

    // AVX chunk: process 8 taps at a time
    for (; j + 8 <= kernelSize; j += 8) 
    {
        // compute the *first* element of the backward window:
        // if j==0 we want samples [nD-0 … nD-7], so base == nD-7
        int base = i * m_DownSampling - j - 7;
        if (base >= 0 && base + 8 <= inputSize) 
        {
            // load x[base..base+7] == x[nD-(j+7) … nD-j]
            __m256 inVals = _mm256_loadu_ps(&input[base]);
            // reverse lanes so lane-k holds x[nD-(j+k)]
            inVals = _mm256_permutevar8x32_ps(inVals, REV_IDX);

            __m256 kVals  = _mm256_loadu_ps(&m_FIRCoefficiants[j]);
            _sum = _mm256_fmadd_ps(inVals, kVals, _sum);
        }
    }

    // scalar tail for any remaining taps
    for (; j < kernelSize; ++j) 
    {
        int idx = i * m_DownSampling - j;
        if (idx >= 0 && idx < inputSize) 
        {
            output[i] += input[idx] * m_FIRCoefficiants[j];
        }
    }

    // horizontal-sum the AVX accumulator into output[i]
    float tmp[8];
    _mm256_storeu_ps(tmp, _sum);
    output[i] += tmp[0] + tmp[1] + tmp[2] + tmp[3]
               + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}
