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

#include "gpu/jit/jit_eltwise_injector.hpp"

#include <limits>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::min_scratch_regs() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd: return 1;
            case eltwise_exp:
            case eltwise_exp_use_dst_for_bwd: return 0;
            case eltwise_gelu_erf: return 4;
            case eltwise_hardswish: return 1;
            case eltwise_log: return 0;
            case eltwise_logsigmoid: return 1;
            case eltwise_mish: return 2;
            case eltwise_pow: return 1;
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd: return (alpha_ == 0.f) ? 0 : 1;
            case eltwise_abs: return 0;
            case eltwise_soft_relu: return 1;
            case eltwise_sqrt:
            case eltwise_sqrt_use_dst_for_bwd: return 0;
            case eltwise_square: return 0;
            case eltwise_swish: return 1;
            case eltwise_tanh:
            case eltwise_tanh_use_dst_for_bwd: return 0;
            case eltwise_round: return 0;
            case eltwise_linear: return 0;
            case eltwise_bounded_relu:
            case eltwise_clip:
            case eltwise_clip_v2:
            case eltwise_clip_v2_use_dst_for_bwd: return 0;
            case eltwise_gelu_tanh: return 2;
            case eltwise_logistic:
            case eltwise_logistic_use_dst_for_bwd: return 0;
            default: assert(!"unsupported eltwise algorithm");
        }
    } else {
        switch (alg_) {
            case eltwise_relu: return 1;
            case eltwise_abs: return 1;
            case eltwise_square: return 0;
            case eltwise_linear: return 0;
            case eltwise_bounded_relu:
            case eltwise_clip: return 1;
            case eltwise_gelu_tanh: return 2;
            default: assert(!"unsupported eltwise algorithm");
        }
    }
    return 0;
}

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::preferred_scratch_regs() {
    using namespace alg_kind;
    if (is_fwd_) {
        switch (alg_) {
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd: return 8;
            case eltwise_gelu_erf: return 8;
            case eltwise_hardswish: return 8;
            case eltwise_mish: return 8;
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd: return (alpha_ == 0.f) ? 0 : 8;
            case eltwise_gelu_tanh: return 8;
            case eltwise_soft_relu: return 8;
            case eltwise_swish: return 8;
            default: break;
        }
    } else {
        switch (alg_) {
            case eltwise_gelu_tanh: return 8;
            default: break;
        }
    }
    return min_scratch_regs();
}

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::max_batch_size() {
    using namespace alg_kind;
    auto ss = scratch_.getLen();

    if (is_fwd_) {
        switch (alg_) {
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd:
                if (alpha_ == 0.)
                    break;
                else
                    return ss;
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd:
            case eltwise_hardswish:
            case eltwise_logsigmoid:
            case eltwise_pow:
            case eltwise_soft_relu:
            case eltwise_swish: return ss;
            case eltwise_mish:
            case eltwise_gelu_erf: return ss / min_scratch_regs();
            case eltwise_gelu_tanh: return ss & ~1;
            default: break;
        }
    } else {
        switch (alg_) {
            case eltwise_gelu_tanh: return ss / 2;
            default: break;
        }
    }

    return 128;
}

template <gpu_gen_t hw>
int jit_eltwise_injector_f32<hw>::phase_count(alg_kind_t alg) {
    using namespace alg_kind;

    if (is_fwd_) {
        switch (alg) {
            case eltwise_elu:
            case eltwise_elu_use_dst_for_bwd: return 5;
            case eltwise_exp:
            case eltwise_exp_use_dst_for_bwd: return 2;
            case eltwise_gelu_erf: return 25;
            case eltwise_hardswish: return 3;
            case eltwise_log: return 2;
            case eltwise_logsigmoid:
                return phase_count(alg_kind::eltwise_soft_relu) + 2;
            case eltwise_mish:
                return phase_count(alg_kind::eltwise_soft_relu)
                        + phase_count(alg_kind::eltwise_tanh) + 1;
            case eltwise_pow: return 6;
            case eltwise_relu:
            case eltwise_relu_use_dst_for_bwd: return (alpha_ == 0) ? 1 : 2;
            case eltwise_soft_relu: return 9;
            case eltwise_swish: return 5;
            case eltwise_tanh:
            case eltwise_tanh_use_dst_for_bwd: return 6;
            case eltwise_linear: return 2;
            case eltwise_bounded_relu:
            case eltwise_clip:
            case eltwise_clip_v2:
            case eltwise_clip_v2_use_dst_for_bwd: return 2;
            case eltwise_gelu_tanh: return 8;
            case eltwise_logistic:
            case eltwise_logistic_use_dst_for_bwd: return 4;
            default: break;
        }
    } else {
        switch (alg) {
            case eltwise_abs: return 2;
            case eltwise_bounded_relu:
            case eltwise_clip: return 4;
            case eltwise_gelu_tanh: return 14;
            default: break;
        }
    }

    return 1;
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_zero_ns_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->max_(simd, r, r, 0.f);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, temp, r, alpha_); break;
        case 1: h->csel(simd | le | f0[0], r, temp, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::abs_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->mov(simd, r, abs(r));
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::soft_relu_compute_fwd_inner(int simd,
        const ngen::GRF &input, const ngen::GRF &temp, const ngen::GRF &dest,
        int phase, int off) {
    const float exp_overflow_bound = 88.72283172607421875;
    const float log2e = 1.44269502162933349609375f;
    const float reciproc_log2e = 1.f / log2e; // 1 / log_2(e)
    switch (phase) {
        case 0: h->mul(simd, temp, input, 1.f); break;
        case 1: h->add(simd, dest, input, -exp_overflow_bound); break;
        case 2: h->csel(simd | le | f0[0], dest, dest, temp, dest); break;
        case 3: h->mul(simd, temp, temp, log2e); break;
        case 4: h->eexp(simd, temp, temp); break;
        case 5: h->add(simd, temp, temp, 1.f); break;
        case 6: h->log(simd, temp, temp); break;
        case 7: h->mul(simd, temp, temp, reciproc_log2e); break;
        case 8: h->csel(simd | le | f0[0], dest, temp, dest, dest); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::soft_relu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    soft_relu_compute_fwd_inner(simd, r, temp, r, phase, off);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::sqrt_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->sqt(simd, r, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::square_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->mul(simd, r, r, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::tanh_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, r, r, -2 * log2e); break;
        case 1: h->exp(simd, r, r); break;
        case 2: h->add(simd, r, r, 1.f); break;
        case 3: h->inv(simd, r, r); break;
        case 4: h->mul(simd, r, r, 2.f); break;
        case 5: h->add(simd, r, r, -1.f); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::round_compute_fwd(
        int simd, const ngen::GRF &r) {
    h->rnde(simd, r, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::swish_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    const float log2e = 1.442695f; // log_2(e)
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, temp, r, -1.f * log2e * alpha_); break;
        case 1: h->exp(simd, temp, temp); break;
        case 2: h->add(simd, temp, temp, 1.f); break;
        case 3: h->inv(simd, temp, temp); break;
        case 4: h->mul(simd, r, r, temp); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::linear_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    switch (phase) {
        case 0: h->mul(simd, r, r, alpha_); break;
        case 1: h->add(simd, r, r, beta_); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::clip_compute_fwd(
        int simd, const ngen::GRF &r, int phase, float alpha, float beta) {
    switch (phase) {
        case 0: h->max_(simd, r, r, alpha); break;
        case 1: h->min_(simd, r, r, beta); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::gelu_tanh_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {

    const float k = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845f; // sqrt(2/pi)
    const float log2e = 1.442695f; // log_2(e)

    int msimd = simd;
    if (hw == gpu_xe_hp)
        msimd = 16; // workaround for intermittent hang with DPAS+EM

    auto a = scratch_[off].f();
    switch (phase) {
        case 0: h->mul(simd, a, r, r); break;
        case 1: h->mul(simd, a, a, k); break;
        case 2: h->mad(simd, a, r, a, r); break;
        case 3: h->mul(simd, a, a, -2 * sqrt_2_over_pi * log2e); break;
        case 4: h->exp(msimd, a, a); break;
        case 5: h->add(simd, a, a, 1.0f); break;
        case 6: h->inv(msimd, a, a); break;
        case 7: h->mul(simd, r, a, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::logistic_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, r, r, -1.f * log2e); break;
        case 1: h->exp(simd, r, r); break;
        case 2: h->add(simd, r, r, 1.f); break;
        case 3: h->inv(simd, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_prepare_bwd() {
    auto neg_slope = scratch_[0].f(0);
    auto pos_slope = scratch_[0].f(4);
    h->mov(1, neg_slope, alpha_);
    h->mov(1, pos_slope, 1.f);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::relu_compute_bwd(
        int simd, const ngen::GRF &r) {
    auto neg_slope = scratch_[0].f(0);
    auto pos_slope = scratch_[0].f(4);
    h->csel(simd | le | f0[0], r, neg_slope, pos_slope, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::abs_prepare_bwd() {
    auto neg_one = scratch_[0].f(0);
    auto pos_one = scratch_[0].f(4);
    h->mov(1, neg_one, -1.f);
    h->mov(1, pos_one, 1.f);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::clip_prepare_bwd() {
    auto pos_inf_imm = Immediate(std::numeric_limits<float>::infinity());
    auto zero = scratch_[0].f(0);
    auto one = scratch_[0].f(1);
    auto pos_inf = scratch_[0].f(2);
    h->mov(1, zero, 0.f);
    h->mov(1, one, 1.f);
    h->mov(1, pos_inf, pos_inf_imm);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::abs_compute_bwd(
        int simd, const ngen::GRF &r, int phase) {
    auto neg_one = scratch_[0].f(0);
    auto pos_one = scratch_[0].f(4);
    switch (phase) {
        case 0: h->csel(simd | lt | f0[0], r, neg_one, r, r); break;
        case 1: h->csel(simd | gt | f0[0], r, pos_one, r, r); break;
        default: break;
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::square_compute_bwd(
        int simd, const ngen::GRF &r) {
    h->add(simd, r, r, r);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::linear_compute_bwd(
        int simd, const ngen::GRF &r) {
    h->mov(simd, r, alpha_);
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::clip_compute_bwd(
        int simd, const ngen::GRF &r, int phase, float alpha, float beta) {
    auto zero = scratch_[0].f(0);
    auto one = scratch_[0].f(1);
    auto pos_inf = scratch_[0].f(2);
    switch (phase) {
        // r[i] = r[i] - alpha
        case 0: h->add(simd, r, r, -alpha); break;
        // r[i] <= 0 => r[i] = infinity
        case 1: h->csel(simd | le | f0[0], r, pos_inf, r, r); break;
        // r[i] = (r[i] + alpha) - beta
        case 2: h->add(simd, r, r, alpha - beta); break;
        // r[i] = (r[i] <= 0 ? 1 : 0)
        case 3: h->csel(simd | le | f0[0], r, one, zero, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::gelu_tanh_compute_bwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {

    const float k = 0.044715f;
    const float sqrt_2_over_pi = 0.7978845f; // sqrt(2/pi)
    const float log2e = 1.442695f; // log_2(e)

    int msimd = simd;
    if (hw == gpu_xe_hp) msimd = 16;

    auto a = scratch_[off].f();
    auto b = scratch_[off + batch].f();
    switch (phase) {
        case 0: h->mul(simd, a, r, r); break;
        case 1: h->mul(simd, b, a, 3.0f * k); break;
        case 2: h->mul(simd, a, a, k); break;
        case 3: h->mad(simd, a, r, a, r); break;
        case 4: h->mad(simd, b, r, b, r); break;
        case 5: h->mul(simd, a, a, -2 * sqrt_2_over_pi * log2e); break;
        case 6: h->mul(simd, b, b, 2 * sqrt_2_over_pi); break;
        case 7: h->exp(msimd, a, a); break;
        case 8: h->add(simd, r, a, 1.0f); break;
        case 9: h->inv(msimd, r, r); break;
        case 10: h->mul(simd, a, a, r); break;
        case 11: h->mul(simd, a, a, b); break;
        case 12: h->add(simd, a, a, 1.0f); break;
        case 13: h->mul(simd, r, r, a); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::elu_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, temp, r, log2e); break;
        case 1: h->exp(simd, temp, temp); break;
        case 2: h->add(simd, temp, temp, -1.f); break;
        case 3: h->mul(simd, temp, temp, alpha_); break;
        case 4: h->csel(simd | le | f0[0], r, temp, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::exp_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float log2e = 1.442695f; // log_2(e)
    switch (phase) {
        case 0: h->mul(simd, r, r, log2e); break;
        case 1: h->exp(simd, r, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::gelu_erf_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {
    auto temp = scratch_[off].f();
    auto at_accum = scratch_[off + batch].f();
    auto tpow = scratch_[off + 2 * batch].f();
    auto temp2 = scratch_[off + 3 * batch].f();
    const float log2e = 1.442695f; // log_2(e)
    const float reciproc_sqrt_2 = 0.707106769084930419921875f; // 1/sqrt(2)
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    switch (phase) {
        case 0: h->mul(simd, temp, abs(r), reciproc_sqrt_2); break;
        case 1: h->mul(simd, temp, temp, p); break;
        case 2: h->add(simd, temp, temp, 1.f); break;
        case 3: h->inv(simd, temp, temp); break;
        case 4: h->mul(simd, at_accum, temp, a1); break;
        case 5: h->mul(simd, tpow, temp, temp); break;
        case 6: h->mul(simd, temp2, tpow, a2); break;
        case 7: h->add(simd, at_accum, temp2, at_accum); break;
        case 8: h->mul(simd, tpow, tpow, temp); break;
        case 9: h->mul(simd, temp2, tpow, a3); break;
        case 10: h->add(simd, at_accum, temp2, at_accum); break;
        case 11: h->mul(simd, tpow, tpow, temp); break;
        case 12: h->mul(simd, temp2, tpow, a4); break;
        case 13: h->add(simd, at_accum, temp2, at_accum); break;
        case 14: h->mul(simd, tpow, tpow, temp); break;
        case 15: h->mul(simd, temp2, tpow, a5); break;
        case 16: h->add(simd, at_accum, temp2, at_accum); break;
        case 17: h->mul(simd, temp, r, r); break;
        case 18: h->mul(simd, temp, temp, -log2e * 0.5f); break;
        case 19: h->exp(simd, temp, temp); break;
        case 20: h->mul(simd, temp, temp, at_accum); break;
        case 21: h->mul(simd, temp, temp, r); break;
        case 22: h->mul(simd, temp, temp, 0.5f); break;
        case 23: h->add(simd, temp2, r, -temp); break;
        case 24: h->csel(simd | le | f0[0], r, temp, temp2, r); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::hardswish_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0: h->add(simd, temp, r, 3.f); break;
        case 1: h->mul(simd | sat, temp, temp, 1.f / 6.f); break;
        case 2: h->mul(simd, r, r, temp); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::log_compute_fwd(
        int simd, const ngen::GRF &r, int phase) {
    const float reciproc_log2e = 1.f / 1.442695f; // 1 / log_2(e)
    switch (phase) {
        case 0: h->log(simd, r, r); break;
        case 1: h->mul(simd, r, r, reciproc_log2e); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::logsigmoid_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    const int srelu_phases = phase_count(alg_kind::eltwise_soft_relu);
    if (phase == 0) h->mov(simd, r, -r);
    if (phase > 0 && phase < srelu_phases + 1)
        soft_relu_compute_fwd(simd, r, phase - 1, off);
    if (phase == srelu_phases + 1) h->mov(simd, r, -r);
    if (phase > srelu_phases + 1) assert(!"invalid phase");
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::mish_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off, int batch) {
    auto temp = scratch_[off].f();
    auto temp2 = scratch_[off + batch].f();
    const int srelu_phases = phase_count(alg_kind::eltwise_soft_relu);
    const int tanh_phases = phase_count(alg_kind::eltwise_tanh);
    if (phase < srelu_phases)
        soft_relu_compute_fwd_inner(simd, r, temp, temp2, phase, off);
    if (phase >= srelu_phases && phase < srelu_phases + tanh_phases)
        tanh_compute_fwd(simd, temp2, phase - srelu_phases);
    if (phase == srelu_phases + tanh_phases) h->mul(simd, r, r, temp2);
    if (phase > srelu_phases + tanh_phases) assert(!"invalid phase");
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::pow_compute_fwd(
        int simd, const ngen::GRF &r, int phase, int off) {
    auto temp = scratch_[off].f();
    switch (phase) {
        case 0:
            if ((long long int)beta_ == beta_) {
                h->mov(simd, temp, abs(r));
            } else {
                h->mov(simd, temp, r);
            }
            break;
        case 1: h->log(simd, temp, temp); break;
        case 2: h->mul(simd, temp, temp, beta_); break;
        case 3: h->exp(simd, temp, temp); break;
        case 4:
            if (((long long int)beta_) & 0x1)
                h->csel(simd | lt | f0[0], temp, -temp, temp, r);
            break;
        case 5: h->mul(simd, r, temp, alpha_); break;
        default: assert(!"invalid phase");
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::compute(const ngen::GRFRange &regs) {
    using namespace alg_kind;

    auto bmax = max_batch_size();
    auto phases = phase_count(alg_);

    for (int idx0 = 0; idx0 < regs.getLen(); idx0 += bmax) {
        auto batch = nstl::min(regs.getLen() - idx0, bmax);

        for (int phase = 0; phase < phases; phase++) {
            for (int ii = 0; ii < batch; ii += 2) {
                int nreg = nstl::min(2, batch - ii);
                int simd = nreg * GRF::bytes(hw) / sizeof(float);
                auto base = regs[idx0 + ii].f();

                if (is_fwd_) {
                    switch (alg_) {
                        case eltwise_elu:
                        case eltwise_elu_use_dst_for_bwd:
                            elu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_exp:
                        case eltwise_exp_use_dst_for_bwd:
                            exp_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_gelu_erf:
                            gelu_erf_compute_fwd(simd, base, phase, ii, batch);
                            break;
                        case eltwise_hardswish:
                            hardswish_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_log:
                            log_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_logsigmoid:
                            logsigmoid_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_mish:
                            mish_compute_fwd(simd, base, phase, ii, batch);
                            break;
                        case eltwise_pow:
                            pow_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_relu:
                        case eltwise_relu_use_dst_for_bwd:
                            if (alpha_ == 0.f)
                                relu_zero_ns_compute_fwd(simd, base);
                            else
                                relu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_bounded_relu:
                            clip_compute_fwd(simd, base, phase, 0, alpha_);
                            break;
                        case eltwise_abs: abs_compute_fwd(simd, base); break;
                        case eltwise_soft_relu:
                            soft_relu_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_sqrt:
                        case eltwise_sqrt_use_dst_for_bwd:
                            sqrt_compute_fwd(simd, base);
                            break;
                        case eltwise_square:
                            square_compute_fwd(simd, base);
                            break;
                        case eltwise_tanh:
                        case eltwise_tanh_use_dst_for_bwd:
                            tanh_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_round:
                            round_compute_fwd(simd, base);
                            break;
                        case eltwise_swish:
                            swish_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_linear:
                            linear_compute_fwd(simd, base, phase);
                            break;
                        case eltwise_clip:
                        case eltwise_clip_v2:
                        case eltwise_clip_v2_use_dst_for_bwd:
                            clip_compute_fwd(simd, base, phase, alpha_, beta_);
                            break;
                        case eltwise_gelu_tanh:
                            gelu_tanh_compute_fwd(simd, base, phase, ii);
                            break;
                        case eltwise_logistic:
                        case eltwise_logistic_use_dst_for_bwd:
                            logistic_compute_fwd(simd, base, phase);
                            break;
                        default: assert(!"unsupported eltwise algorithm");
                    }
                } else {
                    switch (alg_) {
                        case eltwise_relu: relu_compute_bwd(simd, base); break;
                        case eltwise_bounded_relu:
                            clip_compute_bwd(simd, base, phase, 0, alpha_);
                            break;
                        case eltwise_abs:
                            abs_compute_bwd(simd, base, phase);
                            break;
                        case eltwise_square:
                            square_compute_bwd(simd, base);
                            break;
                        case eltwise_linear:
                            linear_compute_bwd(simd, base);
                            break;
                        case eltwise_clip:
                            clip_compute_bwd(simd, base, phase, alpha_, beta_);
                            break;
                        case eltwise_gelu_tanh:
                            gelu_tanh_compute_bwd(simd, base, phase, ii, batch);
                            break;
                        default: assert(!"unsupported eltwise algorithm");
                    }
                }
                // Apply scale.
                if (phase == phases - 1 && scale_ != 1.f) {
                    h->mul(simd, base, base, scale_);
                }
            }
        }
    }
}

template <gpu_gen_t hw>
void jit_eltwise_injector_f32<hw>::prepare() {
    using namespace alg_kind;

    assert(scratch_.getLen() >= min_scratch_regs());

    if (is_fwd_) {
        /* nothing to do */
    } else {
        switch (alg_) {
            case eltwise_relu: relu_prepare_bwd(); break;
            case eltwise_abs: abs_prepare_bwd(); break;
            case eltwise_bounded_relu:
            case eltwise_clip: clip_prepare_bwd(); break;
            default: break;
        }
    }
}

template struct jit_eltwise_injector_f32<gpu_gen9>;
template struct jit_eltwise_injector_f32<gpu_gen11>;
template struct jit_eltwise_injector_f32<gpu_xe_lp>;
template struct jit_eltwise_injector_f32<gpu_xe_hp>;
template struct jit_eltwise_injector_f32<gpu_xe_hpg>;

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
