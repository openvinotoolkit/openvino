// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define cmult(a, b)  ((cfloat)(real(a) * real(b) - imag(a) * imag(b), real(a) * imag(b) + imag(a) * real(b)))
#define crmult(a, b) ((cfloat)(real(a) * (b), imag(a) * (b)))
#define cadd(a, b)   ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define expi(x)      ((cfloat)(cos(x), sin(x)))
#define expmi(x)     ((cfloat)(cos(x), -sin(x)))
#define conj(x)      ((cfloat)(real(x), -imag(x)))
#define czero()      ((cfloat)(0))

KERNEL(stft_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict signal, 
    const __global INPUT1_TYPE* restrict window,
    const __global INPUT2_TYPE* restrict frame_size_buff,
    const __global INPUT3_TYPE* restrict frame_step_buff,
    __global OUTPUT_TYPE* restrict output)
{
    const int freq_id = get_global_id(0);
    const int frame_id = get_global_id(1);
    const int batch = get_global_id(2);

    const int frame_size = (int)frame_size_buff[0];
    const int frame_step = (int)frame_step_buff[0];

    const int window_size = INPUT1_SIZE_X;

    //printf("freq_id: %i, frame_id: %i, batch: %i, frame_size: %i, frame_step: %i, window_size: %i\n", freq_id, frame_id, batch, frame_size, frame_step, window_size );

    printf("INPUT0_SIZE_X: %i\n", INPUT0_SIZE_X);
    const INPUT0_TYPE* restrict signal_for_this_frame = signal + batch*INPUT0_SIZE_X + frame_id*frame_size;
    // FT from def for single freq for given frame:

    cfloat Y = czero();
    const float PI2 = M_PI_F * 2;

    // ay = 2*PI*(k/N) from dft def.
    const float ay = PI2 * freq_id / frame_size;

    

    for(int i = 0; i < frame_size; ++i) {
        const float signal_val = (float)signal_for_this_frame[i];
        const float window_val = (float)window[i];

        const float x_i = signal_val*window_val;

        const cfloat E = expmi(ay*(float)i);

        Y = cadd(Y, crmult(E, x_i));
    }

#if TRANSPOSE_FRAMES
    const int output_real_idx = OUTPUT_GET_INDEX(batch, freq_id, frame_id, 0);
    const int output_imag_idx = OUTPUT_GET_INDEX(batch, freq_id, frame_id, 1);
#else
    const int output_real_idx = OUTPUT_GET_INDEX(batch, frame_id, freq_id, 0);
    const int output_imag_idx = OUTPUT_GET_INDEX(batch, frame_id, freq_id, 1);
#endif

    output[output_real_idx] = (OUTPUT_TYPE)real(Y);
    output[output_imag_idx] = (OUTPUT_TYPE)imag(Y);
}