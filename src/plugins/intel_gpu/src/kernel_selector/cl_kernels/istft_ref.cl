// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Calculates how many values from different frames correspond to the given global output index.
inline int FUNC(calcBinSize)(int globalIdx, int frameSize, int frameStep, int bufferSize) {
    int earlieststartIdx = max(0, globalIdx - frameSize + 1);
    int realstartIdx = ((earlieststartIdx + frameStep - 1) / frameStep) * frameStep;
    int lastPossibleIdx = bufferSize - frameSize;

    if (lastPossibleIdx < realstartIdx)
        return 0;

    int ret = ((min(lastPossibleIdx, globalIdx) - realstartIdx) / frameStep) + 1;
    return ret;
}

inline bool FUNC(isBetween)(int value, int min, int max) {
    return (value >= min && value < max);
}

// Needed to apply zero padding to the window.
inline float FUNC(getWindowVal)(const INPUT1_TYPE* restrict windowBuff, int globalWindowIdx, int windowSize, int frameSize) {
    const int windowPadLeft = (frameSize - windowSize) / 2;
    if (FUNC_CALL(isBetween)(globalWindowIdx, windowPadLeft, windowPadLeft + windowSize)) {
        return (float)windowBuff[globalWindowIdx - windowPadLeft];
    }

    return 0.0f;
}

typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define crmult(a, b) (real(a) * real(b) - imag(a) * imag(b))
#define expi(x)      ((cfloat)(cos(x), sin(x)))
#define conj(x)      ((cfloat)(real(x), -imag(x)))

///////////////////////////////////////////////////////////////////
//
// Basic, reference istft impl.
//
///////////////////////////////////////////////////////////////////
KERNEL(istft_ref)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* restrict signal,
                  const __global INPUT1_TYPE* restrict windowBuff,
                  const __global INPUT2_TYPE* restrict frameSizeBuff,
                  const __global INPUT3_TYPE* restrict frameStepBuff,
#if LENGTH_BUFFER
                  const __global INPUT4_TYPE* restrict lengthBuff,
#endif
                  __global OUTPUT_TYPE* restrict output) {

    const int batch = get_global_id(0);
    const int finalOutputIdxWithinBatch = get_global_id(1);
    const int frameSize = (int)frameSizeBuff[0];
    const int frameStep = (int)frameStepBuff[0];
    const int windowSize = INPUT1_SIZE_X;
    const int freqs = INPUT0_FEATURE_NUM;
    const int DEFAULT_OUTPUT_SIZE = (INPUT0_SIZE_Y - 1) * frameStep + frameSize;

    const int windowPadLeft = (frameSize - windowSize) / 2;

#if LENGTH_BUFFER
    const int wantedLength = (int)lengthBuff[0];
    if (finalOutputIdxWithinBatch >= wantedLength) {
        return;
    }
#endif

    int outputIdxWithinBatch = finalOutputIdxWithinBatch;

#if CENTER
    outputIdxWithinBatch += frameSize / 2;
#endif

    float normalizationSum = 0.0f;
    const int binSize = FUNC_CALL(calcBinSize)(outputIdxWithinBatch, frameSize, frameStep, DEFAULT_OUTPUT_SIZE);
    int startIdx = outputIdxWithinBatch % frameStep;
    while ((outputIdxWithinBatch + (frameSize - startIdx - 1)) >= DEFAULT_OUTPUT_SIZE)
        startIdx += frameStep;

    for (int i = 0; i < binSize; ++i) {
        const int idx = startIdx + i * frameStep;
        const float val = FUNC_CALL(getWindowVal)(windowBuff, idx, windowSize, frameSize);
        normalizationSum += val * val;
    }

    const bool frameSize_even = frameSize % 2 == 0;
    const int freqsHandledInLoop = frameSize_even ? freqs - 1 : freqs;
    float outputVal = 0.0f;

    const int firstFrameID = max(0, (outputIdxWithinBatch - frameSize + 1 + frameStep - 1) / frameStep);
    const int lastFrameID = min(INPUT0_SIZE_Y - 1, outputIdxWithinBatch / frameStep);

    for (int frameID = firstFrameID; frameID <= lastFrameID; ++frameID) {
        const int frameIdxStartWithinBatch = frameID * frameStep;
        const int globalWindowIdx = outputIdxWithinBatch - frameIdxStartWithinBatch;
        const float windowVal = FUNC_CALL(getWindowVal)(windowBuff, globalWindowIdx, windowSize, frameSize);

        if (windowVal == 0.0f) {
            continue;
        }

        // idftPower = 2*PI*(n/N) from idft def.
        const float idftPower = 2.0f * M_PI_F * (float)globalWindowIdx / (float)frameSize;

        float result = signal[INPUT0_GET_INDEX(batch, 0, frameID, 0)];
        // NOTE: since this is real IDFT, we can skip the imaginary part of the first frequency
        for (int freqID = 1; freqID < freqsHandledInLoop; ++freqID) {
            cfloat freqVal_i;
            real(freqVal_i) = signal[INPUT0_GET_INDEX(batch, freqID, frameID, 0)];
            imag(freqVal_i) = signal[INPUT0_GET_INDEX(batch, freqID, frameID, 1)];

            const cfloat e_i = expi(idftPower * (float)(freqID));
            const float val_i = crmult(freqVal_i, e_i);
            result += val_i;

            const cfloat e_i_n = expi(idftPower * (float)(frameSize - freqID));
            const float val_i_n = crmult(conj(freqVal_i), e_i_n);
            result += val_i_n;
        }

        if (frameSize_even) {
            cfloat lastFreq;
            real(lastFreq) = signal[INPUT0_GET_INDEX(batch, freqs - 1, frameID, 0)];
            imag(lastFreq) = signal[INPUT0_GET_INDEX(batch, freqs - 1, frameID, 1)];

            const cfloat e_i = expi(idftPower * (float)(freqs - 1));
            const float val_i = crmult(lastFreq, e_i);
            result += val_i;
        }

        const float finalIRDFTVal = result / frameSize;

        // IRDFT calculation is done.
        // Apply any normalization.
#if NORMALIZED
        const float scale = sqrt((float)frameSize);
#else
        const float scale = 1.0f;
#endif

        outputVal += (finalIRDFTVal * windowVal * scale) / normalizationSum;
    }

    const int globalOutputIdx = OUTPUT_GET_INDEX(0, 0, batch, finalOutputIdxWithinBatch);
    output[globalOutputIdx] = (OUTPUT_TYPE)outputVal;
}

#undef real
#undef imag
#undef crmult
#undef expi
#undef conj
