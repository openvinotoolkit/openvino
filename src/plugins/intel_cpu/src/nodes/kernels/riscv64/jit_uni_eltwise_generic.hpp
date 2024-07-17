// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include <onednn/dnnl.h>
#include "nodes/executors/eltwise.hpp"

#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"

#include "emitters/plugin/riscv64/jit_generator.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "nodes/kernels/jit_eltwise_call_args_ptrs.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

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

struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, jit_generator {
public:
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
    // Scalar architecture specific registers mapping
    // X0     | zero (Hard-wired zero) | [not used]
    // X1     | ra (Return address)    | [not used]
    // X2     | sp (Stack pointer)     | [not used]
    // X3     | gp (Global pointer)    | [not used]
    // X4     | tp (Thread pointer)    | [not used]
    // X5     | t0 (Temporary)         | kernel temporary
    // X6     | t1 (Temporary)         | kernel temporary
    // X7     | t2 (Temporary)         | kernel temporary
    // X8     | s0/fp (Saved register/frame pointer ) | [not used]
    // X9     | s1 (Saved register)                   | [not used]
    // X10    | a0 (Function arguments/return values) | src
    // X11    | a1 (Function arguments/return values) | src
    // X12    | a2 (Function arguments)               | src
    // X13    | a3 (Function arguments)               | src
    // X14    | a4 (Function arguments)               | src
    // X15    | a5 (Function arguments)               | src
    // X16    | a6 (Function arguments)               | src
    // X17    | a7 (Function arguments)               | dst
    // X18    | s2  (Saved registers)   | [not used]
    // X19    | s3  (Saved registers)   | [not used]
    // X20    | s4  (Saved registers)   | [not used]
    // X21    | s5  (Saved registers)   | [not used]
    // X22    | s6  (Saved registers)   | [not used]
    // X23    | s7  (Saved registers)   | [not used]
    // X24    | s8  (Saved registers)   | kernel: aux
    // X25    | s9  (Saved registers)   | kernel: aux
    // X26    | s10 (Saved registers)   | kernel: aux
    // X27    | s11 (Saved registers)   | kernel: tmp <= FIXME: use get_aux_gpr_kernel()
    // X28    | t3 (Temporary)          | aux
    // X29    | t4 (Temporary)          | aux
    // X30    | t5 (Temporary)          | aux
    // X31    | t6 (Temporary)          | aux + kernel: reg_work_amount <= FIXME

    const Reg reg_work_amount = t6;
    const Reg reg_dst = x17;
    const Reg reg_tmp = x27;

    Reg get_src_reg(uint32_t idx);
    Reg get_aux_gpr(const uint32_t idx);
    Reg get_aux_gpr_kernel(const uint32_t idx);

    //VReg vmm_dst {0};

    VReg get_vmm_reg(const uint32_t idx, const LMUL lmul);
    VReg get_dst_vmm(const uint32_t idx, const LMUL lmul);
    VReg get_aux_vmm(const uint32_t idx, const LMUL lmul, const uint32_t start_idx);

    void load_vector(const VReg& data,
                     const Reg& ptr,
                     const ov::element::Type& src_prc,
                     const ov::element::Type& dst_prc,
                     const bool broadcast,
                     const int32_t ptr_offset = 0);

    void store_vector(const Reg& ptr,
                      const VReg& data,
                      const ov::element::Type& src_prc,
                      const ov::element::Type& dst_prc,
                      const int32_t ptr_offset = 0);

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec);

    void compute_eltwise_op(const LMUL lmul, const uint32_t input_reg_count, const uint32_t vmm_dst_idx);
    void apply_post_ops(const LMUL lmul, const uint32_t input_reg_count, const uint32_t vmm_dst_idx);

    uint32_t lmul2int(const LMUL lmul);

    const std::vector<EltwiseData> eltwise_data_;
    const std::vector<ov::intel_cpu::Type> ops_list_;
    dnnl::post_ops post_ops_;

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

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
