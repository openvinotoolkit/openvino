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

    const float windowVal = (float)window[window_id];
    const float windowValPow2 = windowVal * windowVal;
    
    const int frameIdxStart = frame_id * frame_step;

    const int outputIdxWithinBatch =  OUTPUT_GET_INDEX(0, 0, 0, window_id + frameIdxStart);
    const int globalOutputIdx = OUTPUT_GET_INDEX(0, 0, batch, window_id + frameIdxStart);

    float sum = 0.0f;
    const int binSize = calcDivisor(window_id + frameIdxStart, frame_size, frame_step, OUTPUT_SIZE_X);

    int startIDx = window_id%frame_step;
    
    while((outputIdxWithinBatch + (frame_size - startIDx-1)) >= OUTPUT_SIZE_X)
        startIDx += frame_step;

    // for( int i = 0; i < binSize; ++i ) {
    //     const int idx = startIDx + i * frame_step;
    //     const float val = window[idx];
    //     sum += val*val;
    //     //printf("Sum calc: globalOutputIdx: %i, window_id: %i, i: %i, idx: %i, startIDx: %i, sum: %f\n", globalOutputIdx, window_id, i, idx, startIDx, sum);
    // }

    do {
        const float val = window[startIDx];
        sum += val*val;
        startIDx += frame_step;
        //printf("Sum calc: globalOutputIdx: %i, window_id: %i, i: %i, idx: %i, startIDx: %i, sum: %f\n", globalOutputIdx, window_id, binSize, idx, startIDx, sum);
    } while(startIDx < frame_size);

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



    // printf("window_id=%i, real(res)=%f\n", window_id, real(res)/frame_size);

    
    // TODO: handle case when windowVal == 0.0
    //const float finalVAl = finalIRDFTVal / (windowVal*divisor);
    const float finalVAl = (finalIRDFTVal*windowVal) / (sum);

    // TODO: Handle sumation from different frames...(atomics?)
    const OUTPUT_TYPE finalVal = (OUTPUT_TYPE)(finalVAl);

    //*(output+globalOutputIdx)=finalVal;

    const float prev = atomicadd(output+globalOutputIdx, finalVal);

    //printf("globalOutputIdx: %i, finalIRDFTVal: %f, finalVal: %f, divisior: %i, prev: %f, sum: %f\n", globalOutputIdx, finalIRDFTVal, finalVal, binSize, prev, sum);

    //output[OUTPUT_GET_INDEX(0, 0, batch, window_id + frameOffset)] = finalVal;
}