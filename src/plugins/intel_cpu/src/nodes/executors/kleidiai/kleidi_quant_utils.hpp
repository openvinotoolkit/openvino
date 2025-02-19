#ifndef KLEIDI_QUANT_UTILS_HPP
#define KLEIDI_QUANT_UTILS_HPP

#include <iostream>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>


static size_t roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

inline size_t get_num_blocks_per_row(size_t k, size_t bl) {
    return roundup(k, bl) / bl;
}

inline size_t get_rhs_native_stride(size_t x) {
    return roundup(x, 1);
}

inline size_t get_rhs_scale_stride(size_t k, size_t bl) {
    const size_t num_blocks_per_row = get_num_blocks_per_row(k, bl);
    return num_blocks_per_row * sizeof(float);
}

/**
 *  TODO: Some docstring here
 */
void quant_kxn_qs8cx_f32(int64_t n, 
                         int64_t k, 
                         uint32_t bl, 
                         const float* rhs_f32, 
                         int8_t* rhs_qs8cx, 
                         float* rhs_scales)
{
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
    const size_t rhs_qs8cx_stride = get_rhs_native_stride(n);
    std::memset(rhs_qs8cx, 0, k * rhs_qs8cx_stride);

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;
            float max = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const size_t k_idx = block_idx * bl + b;
                if (k_idx >= k) {
                    break;
                }
                const float src0_0 = src_ptr[k_idx];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                    max = src0_0;
                }
            }

            const float scale = max / -128.0;
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            *rhs_scales = scale;
            rhs_scales += 1;

            for (size_t i = 0; i < bl; i++) {
                const size_t k_idx = block_idx * bl + i;
                if (k_idx >= k) {
                    break;
                }
                const float src0_0 = src_ptr[k_idx];

                int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));
                v0_s32 = std::max(INT8_MIN, v0_s32);
                v0_s32 = std::min(INT8_MAX, v0_s32);

                const int8_t v0_i8 = (int8_t)v0_s32;
                const size_t dst_addr = row_idx + k_idx * rhs_qs8cx_stride;
                rhs_qs8cx[dst_addr] = v0_i8;
            }
        }
    }
}

#endif