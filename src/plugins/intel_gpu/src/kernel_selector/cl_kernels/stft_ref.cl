// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// alternative: https://github.com/OpenCL/ComplexMath/blob/master/clcomplex.h
typedef float2 cfloat;
#define real(a)      ((a).s0)
#define imag(a)      ((a).s1)
#define crmult(a, b) ((cfloat)(real(a) * (b), imag(a) * (b)))
#define cadd(a, b)   ((cfloat)(real(a) + real(b), imag(a) + imag(b)))
#define csub(a, b)   ((cfloat)(real(a) - real(b), imag(a) - imag(b)))
#define expmi(x)     ((cfloat)(cos(x), -sin(x)))
#define czero()      ((cfloat)(0))

// Unoptimized, the most obvious stft impl from the definition.
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

    // Handling case where window size is smaller than frame size.
    const int start_offset = (frame_size - window_size) / 2;

    const INPUT0_TYPE* restrict signal_for_this_frame = signal + batch*INPUT0_SIZE_X + frame_id*frame_step + start_offset;

    // FT from def for single freq for given frame:
    cfloat freq_val = czero();

    // dft_power = 2*PI*(k/N) from dft def.
    const float dft_power = 2.0f * M_PI_F * (float)freq_id / (float)frame_size;

    cfloat err = czero();
    for(int i = 0; i < window_size; ++i) {
        const float signal_val = (float)signal_for_this_frame[i];
        const float window_val = (float)window[i];
        const float x_i = signal_val*window_val;
        const cfloat e_i = expmi(dft_power*(float)(i+start_offset));
        const cfloat val_i = crmult(e_i, x_i);

        // Kahan sum algo:
        const cfloat y = csub(val_i, err);
        const cfloat newSum = cadd(freq_val, y);
        err = csub(newSum, freq_val);
        err = csub(err, y);
        freq_val = newSum;
    }

#if TRANSPOSE_FRAMES
    const int output_real_idx = OUTPUT_GET_INDEX(batch, freq_id, frame_id, 0);
    const int output_imag_idx = OUTPUT_GET_INDEX(batch, freq_id, frame_id, 1);
#else
    const int output_real_idx = OUTPUT_GET_INDEX(batch, frame_id, freq_id, 0);
    const int output_imag_idx = OUTPUT_GET_INDEX(batch, frame_id, freq_id, 1);
#endif

    output[output_real_idx] = (OUTPUT_TYPE)real(freq_val);
    output[output_imag_idx] = (OUTPUT_TYPE)imag(freq_val);
}