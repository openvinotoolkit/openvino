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

#ifndef GPU_JIT_JIT_ELTWISE_INJECTOR_HPP
#define GPU_JIT_JIT_ELTWISE_INJECTOR_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gpu/jit/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline bool jit_eltwise_injector_f32_is_supported(alg_kind_t alg) {
    using namespace alg_kind;
    // TODO Enable eltwise_gelu_tanh once accuracy is improved.
    return utils::one_of(alg, eltwise_elu, eltwise_elu_use_dst_for_bwd,
            eltwise_exp, eltwise_exp_use_dst_for_bwd, eltwise_gelu_erf,
            eltwise_hardswish, eltwise_log, eltwise_logsigmoid, eltwise_mish,
            eltwise_pow, eltwise_relu, eltwise_relu_use_dst_for_bwd,
            eltwise_bounded_relu, eltwise_soft_relu, eltwise_sqrt,
            eltwise_sqrt_use_dst_for_bwd, eltwise_square, eltwise_swish,
            eltwise_tanh, eltwise_tanh_use_dst_for_bwd, eltwise_abs,
            eltwise_round, eltwise_linear, eltwise_clip, eltwise_clip_v2,
            eltwise_clip_v2_use_dst_for_bwd, eltwise_logistic,
            eltwise_logistic_use_dst_for_bwd);
}

template <gpu_gen_t hw>
struct jit_eltwise_injector_f32 {
    jit_eltwise_injector_f32(jit_generator<hw> *host, alg_kind_t alg,
            float alpha, float beta, float scale,
            const ngen::GRFRange &scratch = ngen::GRFRange(),
            bool is_fwd = true)
        : alg_(alg)
        , alpha_(alpha)
        , beta_(beta)
        , scale_(scale)
        , is_fwd_(is_fwd)
        , h(host)
        , scratch_(scratch) {

        assert(jit_eltwise_injector_f32_is_supported(alg_));
        assert(scratch_.isEmpty() || (scratch_.getLen() >= min_scratch_regs()));
    }

    int min_scratch_regs();
    int preferred_scratch_regs();
    void set_scratch(const ngen::GRFRange &scratch) { scratch_ = scratch; }

    void prepare();
    void compute(const ngen::GRF &reg) { compute(reg - reg); }
    void compute(const ngen::GRFRange &regs);

private:
    const alg_kind_t alg_;
    const float alpha_;
    const float beta_;
    const float scale_;
    const bool is_fwd_;

    jit_generator<hw> *h;

    ngen::GRFRange scratch_;

    int max_batch_size();
    int phase_count(alg_kind_t alg);

    void relu_prepare_bwd();
    void abs_prepare_bwd();
    void clip_prepare_bwd();

    void relu_zero_ns_compute_fwd(int simd, const ngen::GRF &r);
    void relu_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void abs_compute_fwd(int simd, const ngen::GRF &r);
    void exp_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void elu_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void gelu_erf_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);
    void hardswish_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void log_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void logsigmoid_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void mish_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);
    void pow_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void soft_relu_compute_fwd_inner(int simd, const ngen::GRF &input,
            const ngen::GRF &temp, const ngen::GRF &dest, int phase, int off);
    void soft_relu_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void sqrt_compute_fwd(int simd, const ngen::GRF &r);
    void square_compute_fwd(int simd, const ngen::GRF &r);
    void round_compute_fwd(int simd, const ngen::GRF &r);
    void swish_compute_fwd(int simd, const ngen::GRF &r, int phase, int off);
    void tanh_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void linear_compute_fwd(int simd, const ngen::GRF &r, int phase);
    void clip_compute_fwd(
            int simd, const ngen::GRF &r, int phase, float alpha, float beta);
    void gelu_tanh_compute_fwd(
            int simd, const ngen::GRF &r, int phase, int off);
    void logistic_compute_fwd(int simd, const ngen::GRF &r, int phase);

    void relu_compute_bwd(int simd, const ngen::GRF &r);
    void abs_compute_bwd(int simd, const ngen::GRF &r, int phase);
    void square_compute_bwd(int simd, const ngen::GRF &r);
    void linear_compute_bwd(int simd, const ngen::GRF &r);
    void clip_compute_bwd(
            int simd, const ngen::GRF &r, int phase, float alpha, float beta);
    void gelu_tanh_compute_bwd(
            int simd, const ngen::GRF &r, int phase, int off, int batch);

    const ngen::InstructionModifier le = jit_generator<hw>::le;
    const ngen::InstructionModifier lt = jit_generator<hw>::lt;
    const ngen::InstructionModifier ge = jit_generator<hw>::ge;
    const ngen::InstructionModifier gt = jit_generator<hw>::gt;
    const ngen::InstructionModifier eq = jit_generator<hw>::eq;
    const ngen::InstructionModifier sat = jit_generator<hw>::sat;
    const ngen::FlagRegister f0 = jit_generator<hw>::f0;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_JIT_ELTWISE_INJECTOR_HPP
