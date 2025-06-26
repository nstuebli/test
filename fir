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

----------------------------------------------------

// reflect policy: mirror at [0…N-1]
static inline int reflect_index(int idx, int N) {
    if (idx < 0)      return -idx;
    if (idx >= N)     return 2*(N - 1) - idx;
    return idx;
}

----------------------------------------------------

const int inputSize   = int(input.size());
    const int kernelSize  = int(m_FIRCoefficiants.size());
    const int D           = m_DownSampling;

    // Prepare output (might be pre-sized to inputSize/D)
    // output.assign(output.size(), 0.0f);  
    // but we’ll zero per-sample below

    // precompute the two “cut points” where full‐width AVX is in‐bounds
    int mid1 = (kernelSize - 1 + D - 1) / D;      // ceil((kernelSize-1)/D)
    int mid2 = (inputSize - 1) / D;               // floor((N-1)/D)
    mid1 = std::max(mid1, start);
    mid2 = std::min(mid2, stop);

    // constant index‐reversal for the AVX epilogue sum and the backward window
    const __m256i REV_IDX = _mm256_setr_epi32(7,6,5,4,3,2,1,0);

    // ----------------------------------------------------------
    // 1) Prologue: scalar convolution with reflect
    for (int i = start; i < mid1; ++i)
    {
        float acc = 0.0f;
        int baseSample = i * D;
        for (int j = 0; j < kernelSize; ++j)
        {
            int idx = baseSample - j;
            idx = reflect_index(idx, inputSize);
            acc += input[idx] * m_FIRCoefficiants[j];
        }
        output[i] = acc;
    }

    // ----------------------------------------------------------
    // 2) Middle: branch‐free AVX + FMA + in-register horizontal sum
    for (int i = mid1; i < mid2; ++i)
    {
        // 2a) zero accumulator
        __m256 sum = _mm256_setzero_ps();
        int baseSample = i * D;

        // 2b) vector body, 8 taps at a time, NO inner 'if'
        for (int j = 0; j + 8 <= kernelSize; j += 8)
        {
            // we want samples [baseSample - (j+0) … baseSample - (j+7)]
            // so load the block starting at (baseSample - j - 7) and reverse it
            const int b0 = baseSample - j - 7;
            __m256 inBlk  = _mm256_loadu_ps(&input[b0]);          // unaligned OK
            inBlk          = _mm256_permutevar8x32_ps(inBlk, REV_IDX);
            __m256 kBlk   = _mm256_loadu_ps(&m_FIRCoefficiants[j]);
            sum            = _mm256_fmadd_ps(inBlk, kBlk, sum);
        }

        // 2c) horizontal sum of sum[0..7] → fAcc
        //    sumHi = sum[4..7], sumLo = sum[0..3]
        __m128 lo     = _mm256_castps256_ps128(sum);
        __m128 hi     = _mm256_extractf128_ps(sum, 1);
        __m128 s128   = _mm_add_ps(lo, hi);            // [ s0+s4, s1+s5, s2+s6, s3+s7 ]
        s128          = _mm_hadd_ps(s128, s128);       // [ (s0+s4+s1+s5), (s2+s6+s3+s7), … ]
        s128          = _mm_hadd_ps(s128, s128);       // all lanes = total sum
        float fAcc    = _mm_cvtss_f32(s128);

        // 2d) scalar tail for j % 8 != 0
        int tail = (kernelSize & ~7);
        for (int j = tail; j < kernelSize; ++j) 
        {
            fAcc += input[baseSample - j] * m_FIRCoefficiants[j];
        }

        output[i] = fAcc;
    }

    // ----------------------------------------------------------
    // 3) Epilogue: scalar convolution with reflect
    for (int i = mid2; i < stop; ++i)
    {
        float acc = 0.0f;
        int baseSample = i * D;
        for (int j = 0; j < kernelSize; ++j)
        {
            int idx = baseSample - j;
            idx = reflect_index(idx, inputSize);
            acc += input[idx] * m_FIRCoefficiants[j];
        }
        output[i] = acc;
    }
