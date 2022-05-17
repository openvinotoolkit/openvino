/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/platform.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_layer_normalization_kernels.hpp"
#endif

#include "cpu/simple_layer_normalization_kernels.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace lnorm_utils {

using namespace data_type;

template <>
void stat_and_data_kernel_t<f32>::operator()(const float *src, float *dst,
        const float *scale, const float *shift, float *mean, float *var,
        const size_t block_size) const {
    // XXX: manual unrolling for use_scaleshift_ due to clang issue.
    //      see: CLANG_WA_01_SAFE_TO_USE_OMP_SIMD
    for (size_t offset = 0; offset < block_size; offset++) {
        float v_mean, v_variance;
        if (calculate_stats_) {
            v_mean = 0;
            PRAGMA_OMP_SIMD(reduction(+ : v_mean))
            for (dim_t c = 0; c < C_; ++c) {
                v_mean += src[c + C_ * offset];
            }
            v_mean /= C_;

            v_variance = 0;
            PRAGMA_OMP_SIMD(reduction(+ : v_variance))
            for (dim_t c = 0; c < C_; ++c) {
                const auto src_sub_mean = src[c + C_ * offset] - v_mean;
                v_variance += src_sub_mean * src_sub_mean;
            }
            v_variance /= C_;
        } else {
            v_mean = mean[offset];
            v_variance = var[offset];
        }

        const float inv_sqrtvar = 1. / sqrtf(v_variance + eps_);
        if (use_scaleshift_ || (use_scale_ && use_shift_)) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = scale[c] * inv_sqrtvar;
                const float sv = shift[c];
                const size_t elem = c + C_ * offset;
                dst[elem] = sm * (src[elem] - v_mean) + sv;
            }
        } else if (use_scale_) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = scale[c] * inv_sqrtvar;
                const size_t elem = c + C_ * offset;
                dst[elem] = sm * (src[elem] - v_mean);
            }
        } else if (use_shift_) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = 1.0f * inv_sqrtvar;
                const float sv = shift[c];
                const size_t elem = c + C_ * offset;
                dst[elem] = sm * (src[elem] - v_mean) + sv;
            }
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; ++c) {
                const float sm = 1.0f * inv_sqrtvar;
                const size_t elem = c + C_ * offset;
                dst[elem] = sm * (src[elem] - v_mean);
            }
        }
        if (calculate_stats_ && save_stats_) {
            mean[offset] = v_mean;
            var[offset] = v_variance;
        }
    }
}

template <>
void diff_ss_kernel_t<f32>::operator()(const float *src, const float *diff_dst,
        float *diff_gamma, float *diff_beta, const float *mean,
        const float *var, float *const inv_sqrtvar,
        const size_t block_size) const {
    for (size_t offset = 0; offset < block_size; offset++) {
        inv_sqrtvar[offset] = 1. / sqrtf(var[offset] + eps_);
        PRAGMA_OMP_SIMD()
        for (dim_t c = 0; c < C_; c++) {
            const size_t elem = c + C_ * offset;
            const float dd = diff_dst[elem];
            diff_gamma[c]
                    += (src[elem] - mean[offset]) * dd * inv_sqrtvar[offset];
            diff_beta[c] += dd;
        }
    }
}

template <>
void diff_data_kernel_t<f32>::operator()(const float *src,
        const float *diff_dst, float *diff_src, const float *ss,
        const float *mean, float *const inv_sqrtvar,
        const size_t block_size) const {
    // XXX: manual unrolling for use_scaleshift_ due to clang issue.
    //      see: CLANG_WA_01_SAFE_TO_USE_OMP_SIMD
    float dd_gamma, dd_gamma_x;
    for (size_t offset = 0; offset < block_size; offset++) {
        // reduce gamma
        dd_gamma = dd_gamma_x = 0;
        if (calculate_diff_stats_) {
            if (use_scaleshift_ || use_scale_) {
                PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                for (dim_t c = 0; c < C_; c++) {
                    const size_t elem = c + C_ * offset;
                    const float v_diff_dst = diff_dst[elem];
                    dd_gamma += v_diff_dst * ss[c];
                    dd_gamma_x
                            += v_diff_dst * ss[c] * (src[elem] - mean[offset]);
                }
            } else {
                PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                for (dim_t c = 0; c < C_; c++) {
                    const size_t elem = c + C_ * offset;
                    const float v_diff_dst = diff_dst[elem];
                    dd_gamma += v_diff_dst;
                    dd_gamma_x += v_diff_dst * (src[elem] - mean[offset]);
                }
            }
            dd_gamma_x *= inv_sqrtvar[offset];
        }

        // calculate diff_dst
        if (use_scaleshift_ || use_scale_) {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; c++) {
                const size_t elem = c + C_ * offset;
                float v_diff_src = diff_dst[elem] * ss[c];
                if (calculate_diff_stats_)
                    v_diff_src -= dd_gamma / C_
                            + (src[elem] - mean[offset]) * dd_gamma_x
                                    * inv_sqrtvar[offset] / C_;
                v_diff_src *= inv_sqrtvar[offset];
                diff_src[elem] = v_diff_src;
            }
        } else {
            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C_; c++) {
                const size_t elem = c + C_ * offset;
                float v_diff_src = diff_dst[elem];
                if (calculate_diff_stats_)
                    v_diff_src -= dd_gamma / C_
                            + (src[elem] - mean[offset]) * dd_gamma_x
                                    * inv_sqrtvar[offset] / C_;
                v_diff_src *= inv_sqrtvar[offset];
                diff_src[elem] = v_diff_src;
            }
        }
    }
}

template <>
void diff_data_kernel_t<bf16>::operator()(const bfloat16_t *src,
        const bfloat16_t *diff_dst, bfloat16_t *diff_src, const float *ss,
        const float *mean, float *const inv_sqrtvar,
        const size_t block_size) const {
    assert(!"No default diff_data_kernel_t operator() for bf16 "
            "input!");
}

template <>
void stat_and_data_kernel_t<bf16>::operator()(const bfloat16_t *src,
        bfloat16_t *dst, const float *scale, const float *shift, float *mean,
        float *var, const size_t block_size) const {
    assert(!"No default stat_and_data_kernel_t operator() for bf16 input!");
}

template <>
void diff_ss_kernel_t<bf16>::operator()(const bfloat16_t *src,
        const bfloat16_t *diff_dst, float *diff_gamma, float *diff_beta,
        const float *mean, const float *var, float *const inv_sqrtvar,
        const size_t block_size) const {
    assert(!"No default diff_ss_kernel_t operator() for bf16 input!");
}

// Interface section

template <data_type_t data_type>
stat_and_data_kernel_t<data_type> *stat_and_data_kernel_t<data_type>::create(
        const layer_normalization_pd_t *pd) {
#if DNNL_X64
    if (auto *res
            = x64::lnorm_utils::stat_and_data_kernel_create<data_type>(pd))
        return res;
#endif
    if (data_type == bf16) {
        assert(!"No default stat_and_data_kernel_t for bf16 input!");
        return nullptr;
    }
    return new stat_and_data_kernel_t<data_type>(pd);
}

template <data_type_t data_type>
diff_ss_kernel_t<data_type> *diff_ss_kernel_t<data_type>::create(
        const layer_normalization_pd_t *pd) {
#if DNNL_X64
    if (auto *res = x64::lnorm_utils::diff_ss_kernel_create<data_type>(pd))
        return res;
#endif
    if (data_type == bf16) {
        assert(!"No default diff_ss_kernel_t for bf16 input!");
        return nullptr;
    }
    return new diff_ss_kernel_t<data_type>(pd);
}

template <data_type_t data_type>
diff_data_kernel_t<data_type> *diff_data_kernel_t<data_type>::create(
        const layer_normalization_pd_t *pd) {
#if DNNL_X64
    if (auto *res = x64::lnorm_utils::diff_data_kernel_create<data_type>(pd))
        return res;
#endif
    if (data_type == bf16) {
        assert(!"No default diff_data_kernel_t for bf16 input!");
        return nullptr;
    }
    return new diff_data_kernel_t<data_type>(pd);
}

template struct diff_ss_kernel_t<f32>;
template struct diff_ss_kernel_t<bf16>;
template struct stat_and_data_kernel_t<f32>;
template struct stat_and_data_kernel_t<bf16>;
template struct diff_data_kernel_t<f32>;
template struct diff_data_kernel_t<bf16>;

} // namespace lnorm_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
