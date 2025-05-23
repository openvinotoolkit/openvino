// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_kernel_base.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#    include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#endif  // OPENVINO_ARCH_X86_64

namespace ov::intel_cpu {

enum class NMSBoxEncodeType { CORNER, CENTER };

#if defined(OPENVINO_ARCH_X86_64)

namespace kernel {

struct NmsCompileParams {
    NMSBoxEncodeType box_encode_type;
    bool is_soft_suppressed_by_iou;
};

struct NmsCallArgs {
    const void* selected_boxes_coord[4];
    size_t selected_boxes_num;
    const void* candidate_box;
    const void* iou_threshold;
    void* candidate_status;
    // for soft suppression, score *= scale * iou * iou;
    const void* score_threshold;
    const void* scale;
    void* score;
};

template <dnnl::impl::cpu::x64::cpu_isa_t isa>
class NonMaxSuppression : public JitKernel<NmsCompileParams, NmsCallArgs> {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(NonMaxSuppression)

    explicit NonMaxSuppression(const NmsCompileParams& jcp) : JitKernel(jit_name(), jcp, isa) {}

    void generate() override;

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == dnnl::impl::cpu::x64::avx512_core,
                                                         Xbyak::Zmm,
                                                         isa == dnnl::impl::cpu::x64::avx2,
                                                         Xbyak::Ymm,
                                                         Xbyak::Xmm>::type;
    uint32_t vlen = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen;
    const int vector_step = vlen / sizeof(float);
    const int scalar_step = 1;

    Xbyak::Reg64 reg_boxes_coord0 = r8;
    Xbyak::Reg64 reg_boxes_coord1 = r9;
    Xbyak::Reg64 reg_boxes_coord2 = r10;
    Xbyak::Reg64 reg_boxes_coord3 = r11;
    Xbyak::Reg64 reg_candidate_box = r12;
    Xbyak::Reg64 reg_candidate_status = r13;
    Xbyak::Reg64 reg_boxes_num = r14;
    Xbyak::Reg64 reg_iou_threshold = r15;
    // more for soft
    Xbyak::Reg64 reg_score_threshold = rdx;
    Xbyak::Reg64 reg_score = rbp;
    Xbyak::Reg64 reg_scale = rsi;

    Xbyak::Reg64 reg_load_table = rax;
    Xbyak::Reg64 reg_load_store_mask = rbx;

    // reuse
    Xbyak::Label l_table_constant;
    Xbyak::Reg64 reg_table = rcx;
    Xbyak::Reg64 reg_temp_64 = rdi;
    Xbyak::Reg32 reg_temp_32 = edi;

    const Xbyak::Reg64 reg_params = Xbyak::Reg64(dnnl::impl::cpu::x64::abi_param_regs[0]);

    std::unique_ptr<jit_load_emitter> load_vector_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_scalar_emitter = nullptr;

    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;
    std::vector<size_t> load_pool_gpr_idxs;

    Vmm vmm_boxes_coord0 = Vmm(1);
    Vmm vmm_boxes_coord1 = Vmm(2);
    Vmm vmm_boxes_coord2 = Vmm(3);
    Vmm vmm_boxes_coord3 = Vmm(4);
    Vmm vmm_candidate_coord0 = Vmm(5);
    Vmm vmm_candidate_coord1 = Vmm(6);
    Vmm vmm_candidate_coord2 = Vmm(7);
    Vmm vmm_candidate_coord3 = Vmm(8);
    Vmm vmm_temp1 = Vmm(9);
    Vmm vmm_temp2 = Vmm(10);
    Vmm vmm_temp3 = Vmm(11);
    Vmm vmm_temp4 = Vmm(12);

    Vmm vmm_iou_threshold = Vmm(13);
    Vmm vmm_zero = Vmm(15);

    // soft
    Vmm vmm_score_threshold = Vmm(14);
    Vmm vmm_scale = Vmm(0);

    Xbyak::Opmask k_mask = Xbyak::Opmask(7);
    Xbyak::Opmask k_mask_one = Xbyak::Opmask(6);

    std::shared_ptr<dnnl::impl::cpu::x64::jit_uni_eltwise_injector<isa>> exp_injector;

    inline void hard_nms();

    inline void soft_nms();

    inline void suppressed_by_iou(bool is_scalar);

    inline void suppressed_by_score();

    inline void iou(int ele_num);

    inline void soft_coeff();

    inline void horizontal_mul_xmm(const Xbyak::Xmm& xmm_weight, const Xbyak::Xmm& xmm_aux);

    inline void horizontal_mul();

    inline void prepare_table() {
        auto broadcast_d = [&](int val) {
            for (size_t d = 0; d < vlen / sizeof(int); ++d) {
                dd(val);
            }
        };

        align(64);
        L(l_table_constant);
        broadcast_d(0x3f000000);  // 0.5f
        dw(0x0001);
    }
};

}  // namespace kernel

#endif  // OPENVINO_ARCH_X86_64

}  // namespace ov::intel_cpu
