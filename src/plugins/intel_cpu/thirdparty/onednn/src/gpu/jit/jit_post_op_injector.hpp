/*******************************************************************************
 * Copyright 2021 Intel Corporation
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

#ifndef GPU_JIT_JIT_POST_OP_INJECTOR_HPP
#define GPU_JIT_JIT_POST_OP_INJECTOR_HPP

#include "common/primitive_attr.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/jit/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

inline bool jit_post_op_injector_is_supported(
        const post_ops_t &post_ops, bool skip_sum) {
    bool is_supported = true;
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        const auto &po = post_ops.entry_[idx];
        if (po.is_binary())
            is_supported &= false;
        else if (po.is_convolution())
            is_supported &= false;
        else if (po.is_eltwise())
            is_supported
                    &= jit_eltwise_injector_f32_is_supported(po.eltwise.alg);
        else if (po.is_sum(false, false))
            is_supported &= skip_sum;
    }
    return is_supported;
}

template <gpu_gen_t hw>
struct jit_post_op_injector {
    jit_post_op_injector(jit_generator<hw> *host, data_type_t accumulator_type,
            const post_ops_t &post_ops,
            const ngen::GRFRange &scratch = ngen::GRFRange(),
            bool is_fwd = true)
        : post_ops_(post_ops), is_fwd_(is_fwd), scratch_(scratch) {
        assert(accumulator_type == data_type_t::dnnl_f32);
        workers_.reserve(post_ops.len());
        for (int idx = 0; idx < post_ops.len(); ++idx) {
            const auto &po = post_ops.entry_[idx];
            if (po.is_eltwise())
                workers_.emplace_back(host, po.eltwise.alg, po.eltwise.alpha,
                        po.eltwise.beta, po.eltwise.scale, scratch, is_fwd);
        }
    }

    int min_scratch_regs();
    int preferred_scratch_regs();
    void set_scratch(const ngen::GRFRange &scratch);

    void compute(const ngen::GRF &reg) { compute(reg - reg); }
    void compute(const ngen::GRFRange &regs);

private:
    post_ops_t post_ops_;
    std::vector<jit_eltwise_injector_f32<hw>> workers_;
    bool is_fwd_;
    ngen::GRFRange scratch_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_JIT_JIT_POST_OP_INJECTOR_HPP
