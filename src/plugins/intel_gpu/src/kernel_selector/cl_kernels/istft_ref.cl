// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Enable atomic operations for float and half types
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define atomicAdd(TYPE, a, b)  _atomicAdd(TYPE, a, b)
#define _atomicAdd(TYPE, a, b) atomicAdd_##TYPE(a, b)
#define atomicAdd_float(a, b)  FUNC_CALL(_atomicAdd_float)((volatile global atomic_float*)(a), (b))
#define atomicAdd_half(a, b)   FUNC_CALL(_atomicAdd_half)((volatile global atomic_half*)(a), (b))

// WARNING: It is not possible to use check if atomic_fetch_add for half is available.
// Clang compiler always define __opencl_c_ext_fp16_global_atomic_load_store if cl_khr_fp16
// is enabled. This is a workaround solution.
inline half FUNC(_atomicAdd_half)(volatile atomic_half* address, half value) {
#ifndef __opencl_c_ext_fp16_global_atomic_load_store
#    error "ISTFT requires __opencl_c_ext_fp16_global_atomic_load_store extension"
#endif
    half old = value;
    half orig;
    while ((old = atomic_exchange(address, (orig = atomic_exchange(address, 0)) + old)) != 0)
        ;
    return orig;
}

#ifdef __opencl_c_ext_fp32_local_atomic_add
inline float FUNC(_atomicAdd_float)(volatile atomic_float* address, float value) {
    return atomic_fetch_add(address, value);
}
#else
inline float FUNC(_atomicAdd_float)(volatile atomic_float* address, float value) {
    float old = value;
    float orig;
    while ((old = atomic_exchange(address, (orig = atomic_exchange(address, 0.0f)) + old)) != 0.0f)
        ;
    return orig;
}
#endif
// ------------------------------------------------------------------------------

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
                  volatile __global OUTPUT_TYPE* restrict output) {

    const int batch = get_global_id(0);
    const int frameID = get_global_id(1);
    const int windowIdx = get_global_id(2);
    const int frameSize = (int)frameSizeBuff[0];
    const int frameStep = (int)frameStepBuff[0];
    const int windowSize = INPUT1_SIZE_X;
    const int freqs = INPUT0_FEATURE_NUM;
    const int DEFAULT_OUTPUT_SIZE = (INPUT0_SIZE_Y - 1) * frameStep + frameSize;

    const int windowPadLeft = (frameSize - windowSize) / 2;
    const int globalWindowIdx = windowIdx + windowPadLeft;

#if LENGTH_BUFFER
    const int wantedLength = (int)lengthBuff[0];
#endif

    const float windowVal = FUNC_CALL(getWindowVal)(windowBuff, globalWindowIdx, windowSize, frameSize);
    const int frameIdxStartWithinBatch = frameID * frameStep;

    // Gather normalization sum for current outputIdxWithinBatch.
    const int outputIdxWithinBatch = globalWindowIdx + frameIdxStartWithinBatch;
    float normalizationSum = 0.0f;
    const int binSize = FUNC_CALL(calcBinSize)(outputIdxWithinBatch, frameSize, frameStep, DEFAULT_OUTPUT_SIZE);
    int startIdx = globalWindowIdx % frameStep;
    while ((outputIdxWithinBatch + (frameSize - startIdx - 1)) >= DEFAULT_OUTPUT_SIZE)
        startIdx += frameStep;

    for (int i = 0; i < binSize; ++i) {
        const int idx = startIdx + i * frameStep;
        const float val = FUNC_CALL(getWindowVal)(windowBuff, idx, windowSize, frameSize);
        normalizationSum += val * val;
    }

    // Calculate the IRDFT value for the current globalWindowIdx.
    // idftPower = 2*PI*(n/N) from idft def.
    const float idftPower = 2.0f * M_PI_F * (float)globalWindowIdx / (float)frameSize;

    const bool frameSize_even = frameSize % 2 == 0;
    const int freqsHandledInLoop = frameSize_even ? freqs - 1 : freqs;

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

    const float finalVAl = (finalIRDFTVal * windowVal * scale) / (normalizationSum);
    const OUTPUT_TYPE finalVal = (OUTPUT_TYPE)(finalVAl);

    // Write the result to the output buffer.
    int finalOutputIdxWithinBatch = outputIdxWithinBatch;

#if CENTER
    const int margin = frameSize / 2;
    const int lastIdx = LENGTH_BUFFER ? DEFAULT_OUTPUT_SIZE : DEFAULT_OUTPUT_SIZE - margin;
    if (!FUNC_CALL(isBetween)(finalOutputIdxWithinBatch, margin, lastIdx))
        return;
    finalOutputIdxWithinBatch -= margin;
#endif

#if LENGTH_BUFFER
    if (finalOutputIdxWithinBatch >= wantedLength) {
        return;
    }
#endif

    const int globalOutputIdx = OUTPUT_GET_INDEX(0, 0, batch, finalOutputIdxWithinBatch);

    // Perform last reduction atomically.
    atomicAdd(OUTPUT_TYPE, output + globalOutputIdx, finalVal);
}

#undef real
#undef imag
#undef crmult
#undef expi
#undef conj
#undef atomicAdd
#undef _atomicAdd
#undef atomicAdd_float
#undef atomicAdd_half
