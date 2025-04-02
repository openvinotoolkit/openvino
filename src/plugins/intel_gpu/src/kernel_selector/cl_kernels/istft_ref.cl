// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma OPENCL EXTENSION cl_ext_float_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define atomicAdd(TYPE, a, b)  _atomicAdd(TYPE, a, b)
#define _atomicAdd(TYPE, a, b) atomicAdd_##TYPE(a, b)
#define atomicAdd_float(a, b)  atomic_fetch_add((volatile atomic_float*)(a), (b))
#define atomicAdd_half(a, b)   _atomicAdd_half((volatile atomic_half*)(a), (b))

inline half _atomicAdd_half(volatile atomic_half* address, const half value) {
    half old = value;
    half orig;
    while ((old = atomic_exchange(address, (orig = atomic_exchange(address, 0)) + old)) != 0)
        ;
    return orig;
}

// #else
//   inline float atomicAdd(volatile __global float* address, const float value) {
//     float old = value, orig;
//     while ((old = atomic_xchg(address, (orig = atomic_xchg(address, 0.0f)) + old)) != 0.0f);
//     return orig;
//   }
// #endif

// Calculates how many values correspond to the given global index.
inline int FUNC(calcBinSize)(int globalIdx, int frameSize, int frameStep, int bufferSize) {
    int earliestStartIdx = max(0, globalIdx - frameSize + 1);
    int realStartIdx = ((earliestStartIdx + frameStep - 1) / frameStep) * frameStep;
    int lastPossibleIdx = bufferSize - frameSize;

    if (lastPossibleIdx < realStartIdx)
        return 0;

    int ret = ((min(lastPossibleIdx, globalIdx) - realStartIdx) / frameStep) + 1;
    return ret;
}

inline bool FUNC(isBetween)(int value, int min, int max) {
    return (value >= min && value < max);
}

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define crmult(a, b) (real(a) * real(b) - imag(a) * imag(b))
#define expi(x)      ((cfloat)(cos(x), sin(x)))
#define conj(x)      ((cfloat)(real(x), -imag(x)))

// Unoptimized istft impl.
KERNEL(istft_ref)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* restrict signal,
                  const __global INPUT1_TYPE* restrict window,
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
    const int window_size = INPUT1_SIZE_X;
    const int freqs = INPUT0_FEATURE_NUM;
    const int DEFAULT_OUTPUT_SIZE = (INPUT0_SIZE_Y - 1) * frameStep + frameSize;

#if LENGTH_BUFFER
    const int wantedLength = (int)lengthBuff[0];
#endif

    const float windowVal = (float)window[windowIdx];
    const int frameIdxStartWithinBatch = frameID * frameStep;

    // Gather normalization sum for current windowIdx.
    const int outputIdxWithinBatch = windowIdx + frameIdxStartWithinBatch;
    float normalizationSum = 0.0f;
    const int binSize = FUNC_CALL(calcBinSize)(windowIdx + frameIdxStartWithinBatch, frameSize, frameStep, DEFAULT_OUTPUT_SIZE);
    int startIDx = windowIdx % frameStep;
    while ((outputIdxWithinBatch + (frameSize - startIDx - 1)) >= DEFAULT_OUTPUT_SIZE)
        startIDx += frameStep;

    for (int i = 0; i < binSize; ++i) {
        const int idx = startIDx + i * frameStep;
        const float val = window[idx];
        normalizationSum += val * val;
    }

    // Calculate the IRDFT value for the current windowIdx.
    // idftPower = 2*PI*(n/N) from idft def.
    const float idftPower = 2.0f * M_PI_F * (float)windowIdx / (float)frameSize;

    const bool frameSize_even = frameSize % 2 == 0;
    const int freqsHandledInLoop = frameSize_even ? freqs - 1 : freqs;

    float result = signal[INPUT0_GET_INDEX(batch, 0, frameID, 0)];
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