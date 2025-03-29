// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#define FREQS_PER_THREAD 4

KERNEL(stft_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* restrict signal, 
    const __global INPUT1_TYPE* restrict window,
    const __global INPUT2_TYPE* restrict frame_size_buff,
    const __global INPUT3_TYPE* restrict frame_step_buff,
    __global OUTPUT_TYPE* restrict output)
{
#if TRANSPOSE_FRAMES
    const size_t FREQS = OUTPUT_FEATURE_NUM;
#else
    const size_t FREQS = OUTPUT_SIZE_Y;
#endif 

    const size_t blocksPerFreq = (FREQS + FREQ_PER_BLOCK-1)/FREQ_PER_BLOCK;
    const size_t batch = get_global_id(0);
    const size_t frame_id = get_group_id(1)/blocksPerFreq;
    const size_t freq_start =  (get_group_id(1)%blocksPerFreq)*FREQ_PER_BLOCK;
    const size_t frame_size = (size_t)frame_size_buff[0];
    const size_t frame_step = (size_t)frame_step_buff[0];
    const size_t window_size = INPUT1_SIZE_X;

    __local float x_i_shared[SHARED_X_I_BUFFER_SIZE];

    const size_t block_size = get_local_size(0)*get_local_size(1)*get_local_size(2);

    // Handling case where window size is smaller than frame size.
    const int start_offset = (frame_size - window_size) / 2;

    const INPUT0_TYPE* restrict signal_for_this_frame = signal + batch*INPUT0_SIZE_X + frame_id*frame_step + start_offset;

    // Preload into shared mem:
    for(size_t i = get_local_linear_id()*4; i < window_size; i+= block_size*4) {
        // NOTE: Vectorization by internal unrolling loop, in order to compiler to 
        // decide it if can use vectorized vectorized instructions,
        // which may depend on data type, pointer alignment etc).
        #pragma unroll
        for(size_t j = 0; j < 4; ++j) {
            const float signal_val = (float)signal_for_this_frame[i+j];
            const float window_val = (float)window[i+j];
            x_i_shared[i+j] = signal_val*window_val;
        }
    }

    // Handle leftovers:
    const size_t leftovers_start = window_size%(block_size*4);
    for(size_t i = leftovers_start + get_local_linear_id(); i < window_size; i+= block_size*4) {
        const float signal_val = (float)signal_for_this_frame[i];
        const float window_val = (float)window[i];
        x_i_shared[i] = signal_val*window_val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const size_t max_freq_for_this_block = min(freq_start + FREQ_PER_BLOCK, FREQS);

    // Currently each sub group calcs 4 freq_id at the same time.
    for(size_t freq_id = get_sub_group_id()*FREQS_PER_THREAD + freq_start; freq_id < max_freq_for_this_block; freq_id += get_num_sub_groups()*FREQS_PER_THREAD) {

        float4 freq_val_real = 0.0f;
        float4 freq_val_img = 0.0f;

        // dft_power = 2*PI*(k/N) from dft def.
        float4 dft_power = 2.0f * M_PI_F / (float)frame_size;
        dft_power.s0 *= (float)(freq_id + 0);
        dft_power.s1 *= (float)(freq_id + 1);
        dft_power.s2 *= (float)(freq_id + 2);
        dft_power.s3 *= (float)(freq_id + 3);

        // For bigger window_size kernel is sin cos bound: Probably there is some external 
        // unit to calc sin cos, which is overloaded with commands(each thread issues 8 such instructions).
        // TODO: Implement fft for those cases.
        for(int i = get_sub_group_local_id(); i < window_size; i+= get_sub_group_size()) {
            const float x_i = x_i_shared[i];

            const float4 real = native_cos(dft_power*(float)(i+start_offset))*x_i;
            const float4 img = -native_sin(dft_power*(float)(i+start_offset))*x_i;

            freq_val_real += real;
            freq_val_img += img;
        }

        freq_val_real.s0 = sub_group_reduce_add(freq_val_real.s0);
        freq_val_real.s1 = sub_group_reduce_add(freq_val_real.s1);
        freq_val_real.s2 = sub_group_reduce_add(freq_val_real.s2);
        freq_val_real.s3 = sub_group_reduce_add(freq_val_real.s3);

        freq_val_img.s0 = sub_group_reduce_add(freq_val_img.s0);
        freq_val_img.s1 = sub_group_reduce_add(freq_val_img.s1);
        freq_val_img.s2 = sub_group_reduce_add(freq_val_img.s2);
        freq_val_img.s3 = sub_group_reduce_add(freq_val_img.s3);

        if((freq_id < FREQS) && (get_sub_group_local_id() < 2*min((size_t)FREQS_PER_THREAD, (FREQS - freq_id)))) {
#if TRANSPOSE_FRAMES
            const int output_idx = OUTPUT_GET_INDEX(batch, freq_id + get_sub_group_local_id()/2, frame_id, get_sub_group_local_id() % 2);
#else
            const int output_idx = OUTPUT_GET_INDEX(batch, frame_id, freq_id + get_sub_group_local_id()/2, get_sub_group_local_id() % 2);
#endif
            if ( (get_sub_group_local_id() % 2) == 0)
                output[output_idx] = (OUTPUT_TYPE)freq_val_real[get_sub_group_local_id()/2];
            else 
                output[output_idx] = (OUTPUT_TYPE)freq_val_img[get_sub_group_local_id()/2];
        }
    }
}