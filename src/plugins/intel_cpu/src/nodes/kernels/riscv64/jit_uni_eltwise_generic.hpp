// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_generator.hpp"

#include "cpu_isa_traits.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "nodes/executors/eltwise.hpp"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::riscv64 {

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
struct jit_uni_eltwise_generic : public jit_uni_eltwise_kernel, jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_eltwise_generic)

    jit_uni_eltwise_generic(jit_eltwise_params jep, std::vector<EltwiseData> eltwise_data);
    jit_uni_eltwise_generic() = default;

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

    void generate() override;

    
protected:
    const uint8_t* getCodeAddress() const override {
        return code_section_address;
    }

private:
    // Register mapping in the kernel:
    // x0-x4 | system | not used
    // x5    | temp   | reg_tmp_0 / reg_work_amount
    // x6    | temp   | reg_tmp_1 / reg_vlen
    // x7    | temp   | reg_offsets / reg_loop_step
    // x8    | saved  | reg_bvlen
    // x9    | saved  | dst_gpr
    // x10   | abi    | reg_const_params / emitter_aux_gpr
    // x11   | abi    | reg_indexes / emitter_aux_gpr
    // x12   | abi    | emitter_aux_gpr
    // x13   | abi    | emitter_aux_gpr
    // x14   | abi    | emitter_aux_gpr
    // x15   | abi    | emitter_aux_gpr
    // x16   | abi    | emitter_aux_gpr
    // x17   | abi    | emitter_aux_gpr
    // x18   | saved  | src_gpr
    // x19   | saved  | src_gpr
    // x20   | saved  | src_gpr
    // x21   | saved  | src_gpr
    // x22   | saved  | src_gpr
    // x23   | saved  | src_gpr
    // x24   | saved  | src_gpr
    // x25   | saved  | src_aux_gpr
    // x26   | saved  | src_aux_gpr
    // x27   | saved  | src_aux_gpr
    // x28   | temp   | src_aux_gpr
    // x29   | temp   | src_aux_gpr
    // x30   | temp   | src_aux_gpr
    // x31   | temp   | src_aux_gpr

    inline Xbyak_riscv::Reg dst_gpr() const {
        // x9
        return Xbyak_riscv::Reg(9);
    }

    inline Xbyak_riscv::Reg src_gpr(const int idx) const {
        // x18-24
        OPENVINO_ASSERT(idx >= 0 && idx < MAX_ELTWISE_INPUTS, "src reg " + std::to_string(idx) + " is not supported");
        const auto start = 18;
        return Xbyak_riscv::Reg(start + idx);
    }

    inline Xbyak_riscv::Reg src_aux_gpr(const int idx) const {
        // saved registers: x[18 + input_number]-x[18 + input_number + MAX_ELTWISE_INPUTS], in max case x25-x31
        OPENVINO_ASSERT(idx >= 0 && idx < MAX_ELTWISE_INPUTS, "src aux reg " + std::to_string(idx) + " is not supported");
        const auto start = static_cast<size_t>(src_gpr(0).getIdx()) + jep_.inputs_number;
        return Xbyak_riscv::Reg(start + idx);
    }

    inline Xbyak_riscv::Reg aux_gpr(const int idx) const {
        // saved and temp registers: x[18 + 2 * input_number]-x[31] and then abi x10-x17
        const auto saved_start = src_aux_gpr(0).getIdx() + jep_.inputs_number;
        if (saved_start + idx < gpr_count)
            return Xbyak_riscv::Reg(saved_start + idx);
        const auto abi_start = Xbyak_riscv::a0.getIdx();
        const auto new_idx = idx - (gpr_count - saved_start);
        if (abi_start + new_idx < static_cast<size_t>(src_gpr(0).getIdx()))
            return Xbyak_riscv::Reg(abi_start + new_idx);
        OPENVINO_THROW("Cannot allocate aux register for emitter!");
    }

    inline Xbyak_riscv::FReg aux_fp_gpr(const int idx) const {
        OPENVINO_ASSERT(idx >= 0 && idx < static_cast<int>(fp_gpr_count), "Cannot allocate aux fp register for emitter!");
        return Xbyak_riscv::FReg(idx);
    }

    inline Xbyak_riscv::VReg mask_vec() const {
        return Xbyak_riscv::VReg(0);
    }

    inline Xbyak_riscv::VReg dst_vec() const {
        const auto lmul_v = static_cast<int>(lmul2float(exec_lmul));
        // v0 - mask register
        const auto vec_idx = lmul_v == 0 ? 1 : lmul_v;
        return Xbyak_riscv::VReg(vec_idx);
    }

    inline Xbyak_riscv::VReg src_vec(const int idx) const {
        OPENVINO_ASSERT(idx >= 0 && idx < MAX_ELTWISE_INPUTS, "src aux reg " + std::to_string(idx) + " is not supported");
        const auto lmul_v = static_cast<int>(lmul2float(exec_lmul));
        // v0 and v[lmul] - mask and dst registers
        const auto vec_idx = (idx + 2) * (lmul_v == 0 ? 1 : lmul_v);
        OPENVINO_ASSERT(static_cast<size_t>(vec_idx) < (vec_count - lmul_v + 1),
                        "src vector reg " + std::to_string(vec_idx) + " is not supported");
        return Xbyak_riscv::VReg(vec_idx);
    }

    inline Xbyak_riscv::VReg aux_vec(const int idx = 0) const {
        const auto vstart = src_vec(jep_.inputs_number + 1).getIdx();
        const auto lmul_v = static_cast<int>(lmul2float(exec_lmul));
        const auto vec_idx = vstart + idx * (lmul_v == 0 ? 1 : lmul_v);
        OPENVINO_ASSERT(static_cast<size_t>(vec_idx) < (vec_count - lmul_v + 1),
                        "aux vector reg " + std::to_string(vec_idx) + " is not supported");
        return Xbyak_riscv::VReg(vec_idx);
    }

    inline size_t get_max_aux_fp_gpr_count() const {
        return fp_gpr_count;
    }

    inline size_t get_max_aux_gpr_count() const {
        const auto st_count = gpr_count - (static_cast<size_t>(src_aux_gpr(0).getIdx()) + jep_.inputs_number + 1);
        return st_count + num_abi_param_regs;
    }

    inline size_t get_max_aux_vec_count() const {
        auto lmul_v = static_cast<int>(lmul2float(exec_lmul));
        if (lmul_v == 0) lmul_v = 1;
        const auto single_vec_count = vec_count - (src_vec(0).getIdx() + jep_.inputs_number + 1) * lmul_v;
        return static_cast<size_t>(single_vec_count / lmul_v);
    }

    std::shared_ptr<jit_emitter> create_eltwise_emitter(const EltwiseData& data, const ov::element::Type& exec_prec);

    // Update reg_vlen reg_bvlen using current `work_amount` and target `lmul` and `sew`
    void update_vlen(const Xbyak_riscv::Reg& gpr_work_amount, Xbyak_riscv::SEW sew, Xbyak_riscv::LMUL lmul, bool force = false);

    // Load vector with pointer increment if needed (no broadcasting)
    void load_vector(size_t vec_idx, const Xbyak_riscv::Reg& gpr_ptr, const Xbyak_riscv::Reg& gpr_work_amount,
                     const ov::element::Type& src_prc, const ov::element::Type& dst_prc, bool broadcast);
    // Store vector with pointer increment
    void store_vector(const Xbyak_riscv::Reg& gpr_work_amount, const ov::element::Type& src_prc, const ov::element::Type& dst_prc);

    Xbyak_riscv::LMUL compute_exec_lmul(const ov::element::Type& exec_prc) const;
    Xbyak_riscv::LMUL get_max_lmul(const ov::element::Type& exec_prc) const;

    Xbyak_riscv::Reg reg_const_params = Xbyak_riscv::a0;
    Xbyak_riscv::Reg reg_indexes = Xbyak_riscv::a1;

    Xbyak_riscv::Reg reg_tmp_0 = Xbyak_riscv::t0;
    Xbyak_riscv::Reg reg_tmp_1 = Xbyak_riscv::t1;
    Xbyak_riscv::Reg reg_offsets = Xbyak_riscv::t2;
    Xbyak_riscv::Reg reg_work_amount = reg_tmp_0;
    Xbyak_riscv::Reg reg_vlen = reg_tmp_1;
    Xbyak_riscv::Reg reg_loop_step = reg_offsets;
    Xbyak_riscv::Reg reg_bvlen = Xbyak_riscv::x8; // vlen in bytes

    Xbyak_riscv::LMUL current_lmul = Xbyak_riscv::LMUL::m1;
    Xbyak_riscv::SEW current_sew = Xbyak_riscv::SEW::e32;

    // TODO: Support any LMUL values
    Xbyak_riscv::LMUL exec_lmul = Xbyak_riscv::LMUL::m1;
    Xbyak_riscv::SEW exec_sew = Xbyak_riscv::SEW::e32;

    void compute_eltwise_op() const;
    void apply_post_ops() const;
    void emit_data() const;

    const std::vector<EltwiseData> eltwise_data_;
    const std::vector<ov::intel_cpu::Type> ops_list_;

    std::shared_ptr<jit_emitter> eltwise_emitter = nullptr;
    std::vector<std::shared_ptr<jit_emitter>> post_op_emitters;

    uint8_t* data_section_address = nullptr;
    uint8_t* code_section_address = nullptr;
};

}  // ov::intel_cpu::riscv64
