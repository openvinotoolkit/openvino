// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// #if __OPENCL_C_VERSION__ >= CL_VERSION_3_0
  #pragma OPENCL EXTENSION cl_ext_float_atomics : enable
  #define atomicadd(a,b) atomic_fetch_add((volatile atomic_float *)(a),(b)) 
// #else
//   inline float atomicadd(volatile __global float* address, const float value) {
//     float old = value, orig;
//     while ((old = atomic_xchg(address, (orig = atomic_xchg(address, 0.0f)) + old)) != 0.0f);
//     return orig;
//   }
// #endif

inline int calcDivisor(int idx, int frameSize, int frameStep, int bufferSize) {
    int earliestStartIdx = max(0,idx-frameSize+1);
    int realStartIdx = ((earliestStartIdx+frameStep-1)/frameStep)*frameStep;
    int lastPossibleIdx = bufferSize-frameSize;

    if(lastPossibleIdx<realStartIdx)
        return 0;

    int ret = ((min(lastPossibleIdx,idx)-realStartIdx)/frameStep) + 1;
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

// Unoptimized, the most obvious istft impl from the definition.
// __attribute__((reqd_work_group_size(1, 1, 128)))
// __attribute__((intel_reqd_sub_group_size(16)))
KERNEL(istft_ref)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* restrict signal,
                  const __global INPUT1_TYPE* restrict window,
                  const __global INPUT2_TYPE* restrict frame_size_buff,
                  const __global INPUT3_TYPE* restrict frame_step_buff,
                  volatile __global OUTPUT_TYPE* restrict output) {
    const int batch = get_global_id(0);
    const int frame_id = get_global_id(1);
    const int window_id = get_global_id(2);
    const int frame_size = (int)frame_size_buff[0];
    const int frame_step = (int)frame_step_buff[0];
    const int window_size = INPUT1_SIZE_X;

    const int freqs = INPUT0_FEATURE_NUM;

    // printf("batch=%i, frame_id=%i, window_id=%i, fregs=%i\n", batch, frame_id, window_id, freqs);

    // printf("INPUT0_BATCH_NUM=%i, INPUT0_FEATURE_NUM=%i, INPUT0_SIZE_Y=%i, INPUT0_SIZE_X=%i\n",
    //        INPUT0_BATCH_NUM,
    //        INPUT0_FEATURE_NUM,
    //        INPUT0_SIZE_Y,
    //        INPUT0_SIZE_X);

    const float windowVal = (float)window[window_id];

    // idft_power = 2*PI*(n/N) from idft def.
    const float idft_power = 2.0f * M_PI_F * (float)window_id / (float)frame_size;

    cfloat res;
    real(res) = signal[INPUT0_GET_INDEX(batch, 0, frame_id, 0)];
    imag(res) = signal[INPUT0_GET_INDEX(batch, 0, frame_id, 1)];
    for (int freq_id = 1; freq_id < freqs - 1; ++freq_id) {
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

    cfloat lastFreq;
    real(lastFreq) = signal[INPUT0_GET_INDEX(batch, freqs - 1, frame_id, 0)];
    imag(lastFreq) = signal[INPUT0_GET_INDEX(batch, freqs - 1, frame_id, 1)];

    const cfloat e_i = expi(idft_power * (float)(freqs - 1));
    const cfloat val_i = cmult(lastFreq, e_i);
    res = cadd(res, val_i);

    const float finalIRDFTVal = real(res) / frame_size;

    const int frameIdxStart = frame_id * frame_step;
    const int offset = OUTPUT_GET_INDEX(0, 0, batch, window_id + frameIdxStart);

    // printf("window_id=%i, real(res)=%f\n", window_id, real(res)/frame_size);

    const float divisor = (float)calcDivisor(window_id + frameIdxStart, frame_size, frame_step, OUTPUT_SIZE_X);
    // TODO: handle case when windowVal == 0.0
    const float finalVAl = finalIRDFTVal / (windowVal*divisor);

    // TODO: Handle sumation from different frames...(atomics?)
    const OUTPUT_TYPE finalVal = (OUTPUT_TYPE)(finalVAl);

    // *(output+offset)=finalVal;

    const float prev = atomicadd(output+offset, finalVal);

    // printf("offset: %i, finalIRDFTVal: %f, finalVal: %f, divisior: %f, prev: %f, windowVal: %f\n", offset, finalIRDFTVal, finalVal, divisor, prev, windowVal);

    //output[OUTPUT_GET_INDEX(0, 0, batch, window_id + frameOffset)] = finalVal;
}