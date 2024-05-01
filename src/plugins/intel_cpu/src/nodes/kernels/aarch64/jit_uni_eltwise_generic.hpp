// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <onednn/dnnl.h>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include "nodes/executors/eltwise.hpp"

// TODO: handle x64 headers more accurate and remove undef later
// symbols are defined as global macros as result we should disable them
#undef abi_param1
#undef abi_param2
#undef abi_param3
#undef abi_param4
#undef abi_param5
#undef abi_param6
#undef abi_param7
#undef abi_param8
#undef abi_not_param1

#include <cpu/aarch64/jit_generator.hpp>

#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/plugin/aarch64/jit_eltwise_emitters.hpp"
#include "nodes/kernels/jit_eltwise_call_args_ptrs.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    ov::element::Type src_prc[MAX_ELTWISE_INPUTS];
    ov::element::Type dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
    bool use_runtime_ptrs;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

struct jit_uni_eltwise_kernel {
    void (*ker_)(const node::jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const node::jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes);

    jit_uni_eltwise_kernel() {}
    jit_uni_eltwise_kernel(const jit_eltwise_params& jep) : ker_(nullptr), jep_(jep) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    jit_uni_eltwise_generic(const jit_eltwise_params& jep,
                            const std::vector<EltwiseData>& eltwise_data,
                            const std::vector<ov::intel_cpu::Type>& ops_list,
                            const dnnl::post_ops& post_ops);

    jit_uni_eltwise_generic() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override;

private:
    XReg reg_post_op_ptrs = x10;
    XReg start_to_offsets = reg_post_op_ptrs;

    XReg reg_oc_off = x12;
    XReg reg_const_params = abi_param1;
    XReg reg_indexes = abi_param2;

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    using TRegS = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TRegS;

    // Scalar architecture specific registers mapping
    // aarch64| function     | x64 | function
    // ===========================================
    // X0     | [not used]   | RAX | post_op_ptrs
    // X1     | [not used]   | RBX | dst ptr
    // X2     | [not used]   | RCX | [not used]
    // X3     | [not used]   | RDX | work amount
    // X4     | [not used]   | RDI | [not used]
    // X5     | [not used]   | RSI | d_bias
    // X6     | [not used]   | RBP | d_weights
    // X7     | [not used]   | RSP | <stack pointer>
    // X8     | [not used]   | R8  | src ptr
    // X9     | work amount  | R9  | src ptr
    // X10    | ker temporary| R10 | src ptr
    // X11    | ker temporary| R11 | src ptr
    // X12    | ker temporary (abi_not_param1)   | R12 | src ptr
    // X13    | temporary    | R13 | src ptr
    // X14    | temporary    | R14 | src ptr
    // X15    | temporary    | R15 | temporary
    // X16    | dst
    // X17    | [not used: IP0]
    // X18    | [not used: Apple: The platforms reserve register x18. Don't use this register.]

    // ABI: X19-28: Callee-saved registers
    // X19    | src ptr
    // X20    | src ptr
    // X21    | src ptr
    // X22    | src ptr
    // X23    | kernel used (oneDNN: X_TMP_0)
    // X24    | kernel used (oneDNN: X_TMP_1)
    // X25    | src ptr
    // X26    | src ptr
    // X27    | src ptr
    // X28    | kernel used (oneDNN: X_DEFAULT_ADDR)

    // X29    | [not used: The Frame Pointer (FP)]
    // X30    | [not used: The Link Register (LR)]
    // X31    | [not used: The Stack Pointer (SP)]

    const XReg reg_work_amount = x9;
    const XReg reg_dst = x16;

    inline XReg get_src_reg(uint32_t idx) {
        if (idx > MAX_ELTWISE_INPUTS) {
            OPENVINO_THROW("source vector ptr register " + std::to_string(idx) + " is not supported");
        }

        static const std::vector<uint32_t> src_gprs = { 19, 20, 21, 22, 25, 26, 27 };
        return XReg(src_gprs[idx]);
    }

    inline XReg get_aux_gpr(const uint32_t idx) {
        if (idx > 3) {
            OPENVINO_THROW("aux gpr register " + std::to_string(idx) + " is not supported");
        }

        return XReg(13 + idx);
    }

    // Vector registers mapping
    // A64     | function
    // =======================
    // 00-08   | [not used]
    // 09      | dst
    // 10      | aux
    // 11      | aux
    // 12      | aux
    // 13      | aux
    // 14      | aux
    // 15      | aux
    // 16      | aux
    // 17      | aux
    // 18      | src
    // 19      | src
    // 20      | src
    // 21      | src
    // 22      | src
    // 23      | src
    // 24-31   | [not used]


    TReg vmm_dst {9};

    inline TReg get_vmm_reg(const uint32_t idx) {
        if (idx > MAX_ELTWISE_INPUTS) {
            OPENVINO_THROW("source vector register " + std::to_string(idx) + " is not supported");
        }
        return TReg(18 + idx);
    }

    inline SReg get_scl_reg(const uint32_t idx) {
        if (idx > MAX_ELTWISE_INPUTS) {
            OPENVINO_THROW("source scalar register " + std::to_string(idx) + " is not supported");
        }
        return SReg(18 + idx);
    }

    inline TReg get_aux_vmm(const uint32_t idx) {
        if (idx > 7) {
            OPENVINO_THROW("aux vector register " + std::to_string(idx) + " is not supported");
        }
        return TReg(10 + idx);
    }

    void load_vector(const TReg& data,
                     const XReg& ptr,
                     const ov::element::Type& src_prc,
                     const ov::element::Type& dst_prc,
                     const bool broadcast,
                     const int32_t ptr_offset = 0);

    void load_scalar(const SReg& data,
                     const XReg& ptr,
                     const ov::element::Type& src_prc,
                     const ov::element::Type& dst_prc,
                     const int32_t ptr_offset = 0);

    void store_vector(const XReg& ptr,
                 const TReg& data,
                 const ov::element::Type& src_prc,
                 const ov::element::Type& dst_prc,
                 const int32_t ptr_offset = 0);

    void store_scalar(const XReg& ptr,
                      const SReg& data,
                      const ov::element::Type& src_prc,
                      const ov::element::Type& dst_prc,
                      const int32_t ptr_offset = 0);

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec);

    void compute_eltwise_op();
    void apply_post_ops();

    const std::vector<EltwiseData> eltwise_data_;
    const std::vector<ov::intel_cpu::Type> ops_list_;
    const dnnl::post_ops post_ops_;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters;
};

class eltwise_precision_helper {
public:
    static ov::element::Type get_precision(const size_t inputs_number,
                                           const ov::element::Type (&src_prc)[MAX_ELTWISE_INPUTS],
                                           const std::vector<EltwiseData>& eltwise_data);

private:
    static std::set<std::vector<element::Type>> get_supported_precisions(const Algorithm& algo);
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
