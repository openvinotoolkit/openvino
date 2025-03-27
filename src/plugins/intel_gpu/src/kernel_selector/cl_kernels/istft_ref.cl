// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #if __OPENCL_C_VERSION__ >= CL_VERSION_3_0
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable
#define atomicadd(a, b) atomic_fetch_add((volatile atomic_float*)(a), (b))
// #else
//   inline float atomicadd(volatile __global float* address, const float value) {
//     float old = value, orig;
//     while ((old = atomic_xchg(address, (orig = atomic_xchg(address, 0.0f)) + old)) != 0.0f);
//     return orig;
//   }
// #endif

inline int calcDivisor(int idx, int frameSize, int frameStep, int bufferSize) {
    int earliestStartIdx = max(0, idx - frameSize + 1);
    int realStartIdx = ((earliestStartIdx + frameStep - 1) / frameStep) * frameStep;
    int lastPossibleIdx = bufferSize - frameSize;

    if (lastPossibleIdx < realStartIdx)
        return 0;

    int ret = ((min(lastPossibleIdx, idx) - realStartIdx) / frameStep) + 1;
    return ret;
}

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define cmult(a, b)  ((cfloat)(real(a) * real(b) - imag(a) * imag(b), real(a) * imag(b) + imag(a) * real(b)))
#define crmult(a, b) ((cfloat)(real(a) * (b), imag(a) * (b)))
#define cadd(a, b)   ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define csub(a, b)   ((cfloat)(real(a) - real(b), imag(a) - imag(b)))
#define expi(x)      ((cfloat)(cos(x), sin(x)))
#define expmi(x)     ((cfloat)(cos(x), -sin(x)))
#define czero()      ((cfloat)(0))
#define conj(x)      ((cfloat)(real(x), -imag(x)))

// Unoptimized, the istft impl.
KERNEL(istft_ref)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* restrict signal,
                  const __global INPUT1_TYPE* restrict window,
                  const __global INPUT2_TYPE* restrict frame_size_buff,
                  const __global INPUT3_TYPE* restrict frame_step_buff,
                  volatile __global OUTPUT_TYPE* restrict output) {
    const int batch = get_global_id(0);
    const int frame_id = get_global_id(1);
    const int windowIdx = get_global_id(2);
    const int frame_size = (int)frame_size_buff[0];
    const int frame_step = (int)frame_step_buff[0];
    const int window_size = INPUT1_SIZE_X;
    const int freqs = INPUT0_FEATURE_NUM;
    const int DEFAULT_OUTPUT_SIZE = (INPUT0_SIZE_Y - 1) * frame_step + frame_size;

    const float windowVal = (float)window[windowIdx];
    const float windowValPow2 = windowVal * windowVal;

    const int frameIdxStart = frame_id * frame_step;

    // Gather normalization sum for current windowIdx.
    const int outputIdxWithinBatch = windowIdx + frameIdxStart;
    float normalizationSum = 0.0f;
    const int binSize = calcDivisor(windowIdx + frameIdxStart, frame_size, frame_step, DEFAULT_OUTPUT_SIZE);
    int startIDx = windowIdx % frame_step;
    while ((outputIdxWithinBatch + (frame_size - startIDx - 1)) >= DEFAULT_OUTPUT_SIZE)
        startIDx += frame_step;

    for (int i = 0; i < binSize; ++i) {
        const int idx = startIDx + i * frame_step;
        const float val = window[idx];
        normalizationSum += val * val;
        // printf("normalizationSum calc: globalOutputIdx: %i, windowIdx: %i, i: %i, idx: %i, startIDx: %i, normalizationSum: %f\n", globalOutputIdx, windowIdx,
        // i, idx, startIDx, normalizationSum);
    }

    // Calculate the irDFT value for the current windowIdx.
    // idft_power = 2*PI*(n/N) from idft def.
    const float idft_power = 2.0f * M_PI_F * (float)windowIdx / (float)frame_size;

    const bool frame_size_even = frame_size % 2 == 0;

    const int freqsHandledInLoop = frame_size_even ? freqs - 1 : freqs;

    cfloat res;
    real(res) = signal[INPUT0_GET_INDEX(batch, 0, frame_id, 0)];
    imag(res) = signal[INPUT0_GET_INDEX(batch, 0, frame_id, 1)];
    for (int freq_id = 1; freq_id < freqsHandledInLoop; ++freq_id) {
        cfloat freqVal_i;
        real(freqVal_i) = signal[INPUT0_GET_INDEX(batch, freq_id, frame_id, 0)];
        imag(freqVal_i) = signal[INPUT0_GET_INDEX(batch, freq_id, frame_id, 1)];

        const cfloat e_i = expi(idft_power * (float)(freq_id));
        const cfloat val_i = cmult(freqVal_i, e_i);
        res = cadd(res, val_i);

        const cfloat e_i_n = expi(idft_power * (float)(frame_size - freq_id));
        const cfloat val_i_n = cmult(conj(freqVal_i), e_i_n);
        res = cadd(res, val_i_n);
    }

    if (frame_size_even) {
        cfloat lastFreq;
        real(lastFreq) = signal[INPUT0_GET_INDEX(batch, freqs - 1, frame_id, 0)];
        imag(lastFreq) = signal[INPUT0_GET_INDEX(batch, freqs - 1, frame_id, 1)];

        const cfloat e_i = expi(idft_power * (float)(freqs - 1));
        const cfloat val_i = cmult(lastFreq, e_i);
        res = cadd(res, val_i);
    }

    // Apply any normalization.
    const float finalIRDFTVal = real(res) / frame_size;

#if NORMALIZED
    const float scale = sqrt((float)frame_size);
#else
    const float scale = 1.0f;
#endif
    const float finalVAl = (finalIRDFTVal * windowVal * scale) / (normalizationSum);

    const OUTPUT_TYPE finalVal = (OUTPUT_TYPE)(finalVAl);

#if CENTER
    const int margin = frame_size / 2;
    const int outputIdx = windowIdx + frameIdxStart;
    if (outputIdx < margin || outputIdx > OUTPUT_SIZE_X + margin)
        return;

    const int globalOutputIdx = OUTPUT_GET_INDEX(0, 0, batch, outputIdx - margin);
#else
    const int globalOutputIdx = OUTPUT_GET_INDEX(0, 0, batch, windowIdx + frameIdxStart);
#endif

    // Perform last reduction atomically.
    const float prev = atomicadd(output + globalOutputIdx, finalVal);
}