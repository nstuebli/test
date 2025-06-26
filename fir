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


int base = nD – j – 7;              // first of the 8 you really want
__m256 in = _mm256_loadu_ps(&input[base]); 
// in = [ x[nD–j–7], x[nD–j–6], …, x[nD–j] ]
// reverse lane order so lane 0 holds x[nD–j], …, lane7 holds x[nD–j–7]
const __m256 idx = _mm256_setr_epi32(7,6,5,4,3,2,1,0);
in = _mm256_permutevar8x32_ps(in, idx);
__m256 hblk = _mm256_loadu_ps(&h[j]);
sum = _mm256_fmadd_ps(in, hblk, sum);
