// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

#include <xbyak/xbyak.h>

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"

namespace ov::intel_cpu::x64 {

using namespace dnnl::impl::cpu::x64;

struct jit_args_online_softmax {
    void* data;
    const void* max_past;
    const void* denominator_past;
    void* max;
    void* denominator;
    void* out;
    size_t work_amount_inner;
    size_t work_amount_inner_head_size;
    size_t work_amount_outer;
};

struct jit_params_online_softmax {
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    bool with_calibration = true;
};

// online softmax accept data/max_past/denominator_past and get updated_data/max/denominator/
// local max -> update max -> local denominator -> update denominator -> x/d
// get max and denominator so far with local and past.
// max and denominator is saved and become past in next iteration.
// can not shared as both needed in future. copy new to past after pre ops.
struct jit_uni_online_softmax_kernel {
    void (*ker_)(const jit_args_online_softmax*) = nullptr;

    void operator()(const jit_args_online_softmax* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_online_softmax_kernel(jit_params_online_softmax jcp) : jcp_(jcp) {}
    virtual ~jit_uni_online_softmax_kernel() = default;

    virtual void create_ker() = 0;

    jit_params_online_softmax jcp_;
};

template <cpu_isa_t isa>
struct jit_uni_online_softmax_kernel_f32 : public jit_uni_online_softmax_kernel, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_online_softmax_kernel_f32)

    explicit jit_uni_online_softmax_kernel_f32(jit_params_online_softmax jcp)
        : jit_uni_online_softmax_kernel(jcp),
          jit_generator_t(jit_name()) {}

    void create_ker() override {
        jit_generator_t::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }
    void generate() override;

private:
    // kernel has if? if yes need separate function
    void reduce_xmm(Xbyak::Xmm xmm_val, bool is_max);
    // reduce to lowerest of vmm_idx register
    void reduce_vmm(int vmm_idx, bool is_max);

    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::sse41,
                                                         Xbyak::Xmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Zmm>::type;
    const size_t vlen = cpu_isa_traits_t<isa>::vlen;
    const size_t vector_step = vlen / sizeof(float);

    Xbyak::Reg64 reg_data = r8;
    Xbyak::Reg64 reg_max_past = r9;
    Xbyak::Reg64 reg_denominator_past = r10;
    Xbyak::Reg64 reg_max = r11;
    Xbyak::Reg64 reg_denominator = r12;
    Xbyak::Reg64 reg_work_amount_inner = r13;
    Xbyak::Reg64 reg_work_amount_inner_aux = r15;
    Xbyak::Reg64 reg_work_amount_inner_head_size = rdx;
    Xbyak::Reg64 reg_work_amount_outer = r14;
    Xbyak::Reg64 reg_data_aux = rax;
    Xbyak::Reg64 reg_out = rsi;
    Xbyak::Reg64 reg_params = abi_param1;

    Xbyak::Reg64 reg_aux1 = rbx;

    Xbyak::Label l_table_constant;
    Xbyak::Reg64 reg_table = rbp;

    Vmm vmm_val = Vmm(1);
    Vmm vmm_max = Vmm(2);
    Vmm vmm_denominator = Vmm(3);
    Vmm vmm_max_past = Vmm(4);
    Vmm vmm_denominator_past = Vmm(5);
    Vmm vmm_zero = Vmm(6);

    Xbyak::Xmm xmm_aux1 = Xbyak::Xmm(15);
    Xbyak::Xmm xmm_aux2 = Xbyak::Xmm(14);
    Xbyak::Xmm xmm_aux3 = Xbyak::Xmm(13);
    Vmm vmm_aux1 = Vmm(xmm_aux1.getIdx());
    Vmm vmm_aux2 = Vmm(xmm_aux2.getIdx());
    Vmm vmm_aux3 = Vmm(xmm_aux3.getIdx());

    std::unique_ptr<jit_load_emitter> load_emitter_vector;
    std::unique_ptr<jit_load_emitter> load_emitter_scalar;
    std::unique_ptr<jit_store_emitter> store_emitter_vector;
    std::unique_ptr<jit_store_emitter> store_emitter_scalar;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    std::shared_ptr<jit_uni_eltwise_injector_t<isa>> exp_injector;

void prepare_table() {
    auto broadcast_d = [&](int val) {
        for (size_t d = 0; d < vlen / sizeof(int); ++d) {
            dd(val);
        }
    };

    align(64);
    L(l_table_constant);
    broadcast_d(0xff7fffff);  // float minimum
}

};

}  // namespace ov::intel_cpu::x64