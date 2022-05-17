/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_DEPTHWISE_INJECTOR_HPP
#define CPU_X64_JIT_UNI_DEPTHWISE_INJECTOR_HPP

#include <assert.h>

#include "../../../common/c_types_map.hpp"
#include "../../../common/primitive_attr.hpp"
#include "../../../common/type_helpers.hpp"
#include "../../../common/utils.hpp"

#include "../jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace depthwise_injector {

struct static_params_t {
    static_params_t(int vmm_d_weights_idx = 0, int vmm_d_bias_idx = 0,
                    Xbyak::Reg64 reg_d_weights = Xbyak::Reg64(0), Xbyak::Reg64 reg_d_bias = Xbyak::Reg64(0)) :
        vmm_d_weights_idx(vmm_d_weights_idx), vmm_d_bias_idx(vmm_d_bias_idx), reg_d_weights(reg_d_weights), reg_d_bias(reg_d_bias) {}

    int vmm_d_weights_idx;
    int vmm_d_bias_idx;
    Xbyak::Reg64 reg_d_weights;
    Xbyak::Reg64 reg_d_bias;
};

struct dynamic_params_t {
    dynamic_params_t(int vmm_d_weights_idx = 0, int vmm_d_bias_idx = 0,
                     Xbyak::Reg64 reg_d_weights = Xbyak::Reg64(0), Xbyak::Reg64 reg_d_bias = Xbyak::Reg64(0),
                     Xbyak::Reg64 reg_init_off = Xbyak::Reg64(0), const std::map<size_t, int> vmm_idx_off = {},
                     Xbyak::Reg64 reg_post_ops_data = Xbyak::Reg64(0), int base_post_ops_data_offset = 0) :
            vmm_d_weights_idx(vmm_d_weights_idx), vmm_d_bias_idx(vmm_d_bias_idx), reg_d_weights(reg_d_weights), reg_d_bias(reg_d_bias),
            reg_init_off(reg_init_off), reg_init_off_addr(0), vmm_idx_off(vmm_idx_off), useAddr(false),
            reg_post_ops_data(reg_post_ops_data), base_post_ops_data_offset(base_post_ops_data_offset) {}

    dynamic_params_t(int vmm_d_weights_idx, int vmm_d_bias_idx,
                     Xbyak::Reg64 reg_d_weights, Xbyak::Reg64 reg_d_bias,
                     Xbyak::Address reg_init_off, const std::map<size_t, int> vmm_idx_off,
                     Xbyak::Reg64 reg_post_ops_data = Xbyak::Reg64(0), int base_post_ops_data_offset = 0) :
            vmm_d_weights_idx(vmm_d_weights_idx), vmm_d_bias_idx(vmm_d_bias_idx), reg_d_weights(reg_d_weights), reg_d_bias(reg_d_bias),
            reg_init_off(0), reg_init_off_addr(reg_init_off), vmm_idx_off(vmm_idx_off), useAddr(true),
            reg_post_ops_data(reg_post_ops_data), base_post_ops_data_offset(base_post_ops_data_offset) {}

    int vmm_d_weights_idx;
    int vmm_d_bias_idx;
    Xbyak::Reg64 reg_d_weights;
    Xbyak::Reg64 reg_d_bias;
    Xbyak::Reg64 reg_init_off;
    Xbyak::Address reg_init_off_addr;
    std::map<size_t, int> vmm_idx_off;
    bool useAddr;
    Xbyak::Reg64 reg_post_ops_data;
    int base_post_ops_data_offset;
};

} // quantization_injector

template <cpu_isa_t isa>
struct jit_uni_depthwise_injector_f32 {
    using Vmm = typename utils::conditional3<isa == sse41, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    jit_uni_depthwise_injector_f32(jit_generator* host, dnnl_post_ops::entry_t post_op, Xbyak::Opmask k_mask_ = Xbyak::Opmask(1))
            : h(host), post_op_(post_op), k_mask(k_mask_) {
        depthwise_alg = post_op.depthwise.alg;
        assert(utils::one_of(depthwise_alg, alg_kind::depthwise_scale_shift, alg_kind::depthwise_prelu));
    }

    void compute_vector_range(int start_idx, int end_idx, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false);

    void init_ptrs(const Xbyak::RegExp& ptr_data,
                   const Xbyak::Reg64& reg_d_weights, const Xbyak::Reg64& reg_d_bias,
                   const Xbyak::Operand& ch_off, bool is_broadcast);

    void compute(int start_idx, int end_idx,
                 int vmm_d_weights_idx, int vmm_d_bias_idx,
                 const Xbyak::Reg64& reg_d_weights, const Xbyak::Reg64& reg_d_bias,
                 bool is_broadcast = false, int offset = 0, bool need_to_preserve = false);

    static constexpr size_t memoryStep() {
        return sizeof(float*);
    }

private:
    jit_generator* h;

    size_t vlen = cpu_isa_traits<isa>::vlen;

    alg_kind_t depthwise_alg;

    mutable Vmm vmm_mask;
    mutable Vmm vmm_aux0;

    dnnl_post_ops::entry_t post_op_;

    Xbyak::Opmask k_mask;

    const static size_t preserved_vecs_max = 5;
    size_t vecs_to_preserve = 0;
    size_t vecs_count = isa == avx512_common ? 32 : 16;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t start_idx_tail = 0;

    int aux_vecs_count(alg_kind_t elt_alg, bool is_broadcast);

    void compute_body(size_t start_idx, size_t end_idx, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false);
    void injector_preamble(size_t start_idx, size_t end_idx, bool is_broadcast = false);
    void injector_preamble_tail(size_t start_idx, size_t end_idx);
    void injector_postamble();
    void assign_regs();

    void scale_shift_compute_vector(const Vmm &vmm_src, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false, int offset = 0);
    void prelu_compute_vector(const Vmm &vmm_src, const Xbyak::Reg64& p_weights, const Xbyak::Reg64& p_bias, bool is_broadcast = false, int offset = 0);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
