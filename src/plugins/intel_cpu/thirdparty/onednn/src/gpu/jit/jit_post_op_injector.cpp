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

#include "gpu/jit/jit_post_op_injector.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

using namespace ngen;

template <gpu_gen_t hw>
int jit_post_op_injector<hw>::min_scratch_regs() {
    int regs_cnt = 0;
    for (size_t idx = 0; idx < workers_.size(); ++idx) {
        regs_cnt = nstl::max(regs_cnt, workers_[idx].min_scratch_regs());
    }
    return regs_cnt;
}

template <gpu_gen_t hw>
int jit_post_op_injector<hw>::preferred_scratch_regs() {
    int regs_cnt = 0;
    for (size_t idx = 0; idx < workers_.size(); ++idx) {
        regs_cnt = nstl::max(regs_cnt, workers_[idx].preferred_scratch_regs());
    }
    return regs_cnt;
}

template <gpu_gen_t hw>
void jit_post_op_injector<hw>::set_scratch(const ngen::GRFRange &scratch) {
    for (size_t idx = 0; idx < workers_.size(); ++idx) {
        workers_[idx].set_scratch(scratch);
    }
    scratch_ = scratch;
}

template <gpu_gen_t hw>
void jit_post_op_injector<hw>::compute(const ngen::GRFRange &regs) {
    for (size_t idx = 0; idx < workers_.size(); ++idx) {
        workers_[idx].compute(regs);
    }
}

template struct jit_post_op_injector<gpu_gen9>;
template struct jit_post_op_injector<gpu_gen11>;
template struct jit_post_op_injector<gpu_xe_lp>;
template struct jit_post_op_injector<gpu_xe_hp>;
template struct jit_post_op_injector<gpu_xe_hpg>;

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
