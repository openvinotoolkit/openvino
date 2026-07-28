/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#ifdef _WIN32
#include "../../../common/utils/cm/common.hpp"
#else
#include "common/utils/cm/common.hpp"
#endif

namespace gpu::xetla {

/// @addtogroup xetla_util_rand
/// @{

template <uint32_t SIMD = 16, uint32_t round = 7>
struct xetla_rand_t {
    xetla_vector<uint32_t, 2 * SIMD> key;
    xetla_vector<uint32_t, 4 * SIMD> counter;
    static constexpr uint32_t kPhilox10A = 0x9E3779B9;
    static constexpr uint32_t kPhilox10B = 0xBB67AE85;
    static constexpr uint32_t kPhiloxSA = 0xD2511F53;
    static constexpr uint32_t kPhiloxSB = 0xCD9E8D57;

    __XETLA_API void init(uint64_t seed, uint64_t subseq, uint64_t offset) {
        xetla_vector<uint64_t, 1> seed_v = seed;
        xetla_vector<uint64_t, 1> offset_v = offset;
        xetla_vector<uint64_t, 1> subseq_v = subseq * SIMD;

        auto key_2d = key.xetla_format<uint32_t, 2, SIMD>();
        key_2d.row(0) = uint32_t(seed_v.xetla_format<uint32_t>()[0]);
        key_2d.row(1) = uint32_t(seed_v.xetla_format<uint32_t>()[1]);

        xetla_vector<uint32_t, SIMD> channel_id
                = xetla_vector_gen<uint32_t, SIMD>(0, 1);
        auto counter_2d = counter.xetla_format<uint32_t, 4, SIMD>();
        counter_2d.row(0) = uint32_t(offset_v.xetla_format<uint32_t>()[0]);
        counter_2d.row(1) = uint32_t(offset_v.xetla_format<uint32_t>()[1]);
        counter_2d.row(2) = uint32_t(subseq_v.xetla_format<uint32_t>()[0]);
        counter_2d.row(3) = uint32_t(subseq_v.xetla_format<uint32_t>()[1]);
        counter_2d.row(2) += channel_id;
    }

    __XETLA_API xetla_vector<uint32_t, 4 * SIMD> rand() {
        xetla_vector<uint32_t, 2 *SIMD> key_ = key;
        xetla_vector<uint32_t, 4 *SIMD> counter_ = counter;
        auto key_2d_ = key_.xetla_format<uint32_t, 2, SIMD>();

#pragma unroll
        for (int i = 0; i < round; i++) {
            counter_ = single_round(counter_, key_);
            key_2d_.row(0) += kPhilox10A;
            key_2d_.row(1) += kPhilox10B;
        }
        xetla_vector<uint32_t, 4 *SIMD> output = single_round(counter_, key_);
        incr();
        return output;
    }

private:
    __XETLA_API xetla_vector<uint32_t, 4 * SIMD> single_round(
            xetla_vector<uint32_t, 4 * SIMD> counter_,
            xetla_vector<uint32_t, 2 * SIMD> key_) {
        xetla_vector<uint32_t, 4 * SIMD> ret;
        auto ret_2d = ret.xetla_format<uint32_t, 4, SIMD>();
        auto key_2d_ = key_.xetla_format<uint32_t, 2, SIMD>();
        auto counter_2d_ = counter_.xetla_format<uint32_t, 4, SIMD>();

        xetla_vector<uint32_t, SIMD> res0_lo;
        xetla_vector<uint32_t, SIMD> res1_lo;
        xetla_vector<uint32_t, SIMD> res0_hi
                = xetla_imul<uint32_t, uint32_t, uint32_t, SIMD>(
                        res0_lo.xetla_format<uint32_t>(), counter_2d_.row(0),
                        kPhiloxSA);
        xetla_vector<uint32_t, SIMD> res1_hi
                = xetla_imul<uint32_t, uint32_t, uint32_t, SIMD>(
                        res1_lo.xetla_format<uint32_t>(), counter_2d_.row(2),
                        kPhiloxSB);

        ret_2d.row(0) = res1_hi ^ counter_2d_.row(1) ^ key_2d_.row(0);
        ret_2d.row(1) = res1_lo;
        ret_2d.row(2) = res0_hi ^ counter_2d_.row(3) ^ key_2d_.row(1);
        ret_2d.row(3) = res0_lo;

        return ret;
    }

    __XETLA_API void incr() {
        auto counter_2d = counter.xetla_format<uint32_t, 4, SIMD>();
        xetla_vector<uint32_t, SIMD> carry;

        counter_2d.row(0) = xetla_add_c<uint32_t, SIMD>(
                counter_2d.row(0), 1, carry.xetla_format<uint32_t>());
        counter_2d.row(1) = xetla_add_c<uint32_t, SIMD>(
                counter_2d.row(1), carry, carry.xetla_format<uint32_t>());
        counter_2d.row(2) = xetla_add_c<uint32_t, SIMD>(
                counter_2d.row(2), carry, carry.xetla_format<uint32_t>());
        counter_2d.row(3) += carry;
    }
};

template <uint32_t SZ, typename dtype_mask = uint8_t, uint32_t random_simd = 16>
struct dropout_fwd_t {
    static constexpr uint32_t random_len = 4 * random_simd;
    xetla_rand_t<random_simd> rand_gen;
    xetla_vector<dtype_mask, SZ> mask;
    uint32_t threshold;
    float scale;

    __XETLA_API void init(uint64_t seed, uint64_t subseq, uint64_t offset,
            uint32_t threshold_, float scale_) {
        rand_gen.init(seed, subseq, offset);
        this->threshold = threshold_;
        this->scale = scale_;
    }

    template <typename dtype>
    __XETLA_API xetla_vector<dtype, SZ> process(xetla_vector<dtype, SZ> input) {
        xetla_vector<dtype, SZ> output = input;
#pragma unroll
        for (int i = 0; i < SZ / random_len; i++) {
            auto out_sub = output.xetla_select<random_len, 1>(i * random_len);
            auto mask_sub = mask.xetla_select<random_len, 1>(i * random_len);
            xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
            xetla_mask<random_len> mask_flag = rand_val < threshold;
            out_sub.xetla_merge(0, mask_flag);
            mask_sub.xetla_merge(1, 0, mask_flag);
            out_sub = out_sub * scale;
        }
        if constexpr (SZ % random_len != 0) {
            constexpr uint32_t remain_len = SZ % random_len;
            constexpr uint32_t remain_start = SZ / random_len * random_len;
            auto out_sub = output.xetla_select<remain_len, 1>(remain_start);
            auto mask_sub = mask.xetla_select<remain_len, 1>(remain_start);
            // dropout, still generate random_len
            xetla_vector<uint32_t, random_len> rand_val = rand_gen.rand();
            xetla_mask<random_len> mask_flag = rand_val < threshold;
            out_sub.xetla_merge(0, mask_flag.xetla_select<remain_len, 1>(0));
            mask_sub.xetla_merge(
                    1, 0, mask_flag.xetla_select<remain_len, 1>(0));
            out_sub = out_sub * scale;
        }
        return output;
    }

    __XETLA_API xetla_vector<dtype_mask, SZ> get_mask() {
        return mask;
    }
};

/// @} xetla_util_rand

} // namespace gpu::xetla
