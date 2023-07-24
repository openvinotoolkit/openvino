// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <onednn/dnnl.h>
#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>

#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include <emitters/aarch64/jit_emitter.hpp>
#include <emitters/aarch64/jit_eltwise_emitters.hpp>
#include "nodes/kernels/jit_eltwise_call_args_ptrs.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;
using namespace InferenceEngine;

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

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

struct EltwiseData {
    Algorithm algo;
    dnnl::algorithm onednnAlgorithm;
    float alpha;
    float beta;
    float gamma;

    bool operator==(const EltwiseData& rhs) const noexcept {
        return algo == rhs.algo &&
            onednnAlgorithm == rhs.onednnAlgorithm &&
            alpha == rhs.alpha &&
            beta == rhs.beta &&
            gamma == rhs.gamma;
    }
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
    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    using TRegS = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TRegS;

    // Scalar architecture specific registers mapping
    //
    // aarch64| function     | x64 | function
    // ===========================================
    // X0     | <abi param>  | RAX | post_op_ptrs
    // X1     | <abi param>  | RBX | dst ptr
    // X2     | <abi param>  | RCX | [not used]
    // X3     | <abi param>  | RDX | work amount
    // X4     | <abi param>  | RDI | [not used]
    // X5     | <abi param>  | RSI | d_bias
    // X6     | <abi param>  | RBP | d_weights
    // X7     | <abi param>  | RSP | <stack pointer>
    // X8     | [not used]   | R8  | src ptr
    // X9     | work amount  | R9  | src ptr
    // X10    | dst ptr      | R10 | src ptr
    // X11    | src ptr      | R11 | src ptr
    // X12    | src ptr      | R12 | src ptr
    // X13    | src ptr      | R13 | src ptr
    // X14    | src ptr      | R14 | src ptr
    // X15    | src ptr      | R15 | temporary
    // X16    | src ptr
    // X16    | src ptr
    // X17    | temporary
    // X18    | temporary
    // X19-30 | [not used]

    const XReg reg_work_amount = x9;
    const XReg reg_dst = x10;

    inline XReg get_src_reg(uint32_t idx) {
        if (idx > MAX_ELTWISE_INPUTS) {
            IE_THROW(Unexpected) << "source vector ptr register " << idx << " is not supported";
        }
        return XReg(11 + idx);
    }

    // Vector registers mapping
    //
    // A64/X64 | function
    // =======================
    // 0       | [not used]
    // 01      | srs
    // 02      | srs
    // 03      | srs
    // 04      | srs
    // 05      | srs
    // 06      | srs
    // 07      | srs
    // 08      | srs
    // 09      | dst
    // 10      | aux
    // 11      | aux
    // 12      | d_weights
    // 13      | d_bias
    // 14      | [not used]
    // 15      | zero
    // 16 - 30 | [not used]

    TReg vmm_dst {9};

    inline TReg get_vmm_reg(const uint32_t idx) {
        if (idx > MAX_ELTWISE_INPUTS) {
            IE_THROW(Unexpected) << "source vector register " << idx << " is not supported";
        }
        return TReg(1 + idx);
    }

    inline SReg get_scl_reg(const uint32_t idx) {
        if (idx > MAX_ELTWISE_INPUTS) {
            IE_THROW(Unexpected) << "source scalar register " << idx << " is not supported";
        }
        return SReg(1 + idx);
    }

    inline TReg get_aux_vmm(const uint32_t idx) {
        if (idx > 2) {
            IE_THROW(Unexpected) << "aux vector register " << idx << " is not supported";
        }
        return TReg(10 + idx);
    }

    inline XReg get_aux_gpr(const uint32_t idx) {
        if (idx > 2) {
            IE_THROW(Unexpected) << "aux gpr register " << idx << " is not supported";
        }
        return XReg(17 + idx);
    }

    void uni_ldr(const TReg& data,
                 const XReg& ptr,
                 const Precision& src_prc,
                 const Precision& dst_prc,
                 const bool broadcast,
                 const int32_t offset = 0);

    void uni_ldr(const SReg& data,
                 const XReg& ptr,
                 const Precision& src_prc,
                 const Precision& dst_prc,
                 const int32_t offset = 0);

    void uni_str(const XReg& ptr,
                 const TReg& data,
                 const Precision& src_prc,
                 const Precision& dst_prc,
                 const int32_t offset = 0);

    void uni_str(const XReg& ptr,
                 const SReg& data,
                 const Precision& src_prc,
                 const Precision& dst_prc,
                 const int32_t offset = 0);

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, const Precision& exec_prec);

    void compute_eltwise_op();
    void apply_post_ops();

    const std::vector<EltwiseData> eltwise_data_;
    const std::vector<ov::intel_cpu::Type> ops_list_;
    const dnnl::post_ops post_ops_;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
