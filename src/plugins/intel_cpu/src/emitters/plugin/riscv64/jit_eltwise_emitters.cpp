// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include "transformations/cpu_opset/common/op/leaky_relu.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

namespace {
ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(input.get_source_output().get_element_type());
    }

    assert(std::all_of(input_precisions.begin(),
                       input_precisions.end(),
                       [&input_precisions](const ov::element::Type& precision) {
                           return precision == input_precisions[0];
                       }));

    return input_precisions[0];
}
}  // namespace


/// ADD ///
jit_add_emitter::jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_add_emitter::jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_add_emitter::get_inputs_num() const {
    return 2;
}

void jit_add_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfadd_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}


/// Clamp ///
jit_clamp_emitter::jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {
    if (const auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(node)) {
        min = static_cast<float>(clamp->get_min());
        max = static_cast<float>(clamp->get_max());
    } else {
        OPENVINO_THROW("Incompatible node!");
    }
}

jit_clamp_emitter::jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, float min, float max, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc), min(min), max(max) {}

size_t jit_clamp_emitter::get_inputs_num() const {
    return 1;
}

void jit_clamp_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    FReg fmin = f0, fmax = f0;

    h->flw(fmin, p_table, 0);
    h->vfmax_vf(dst, src, fmin);

    h->flw(fmax, p_table, sizeof(float));
    h->vfmin_vf(dst, dst, fmin);
}

std::set<std::vector<element::Type>> jit_clamp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

bool jit_clamp_emitter::need_table() const {
    return true;
}

const void* jit_clamp_emitter::get_table() const {
    static float tbl[2];
    tbl[0] = min; // use explicit assignment to change dynamically array in runtime
    tbl[1] = max;
    return tbl;
}

/// DIV ///
jit_div_emitter::jit_div_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_div_emitter::jit_div_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_div_emitter::get_inputs_num() const {
    return 2;
}

void jit_div_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfdiv_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_div_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// Exp ///
jit_exp_emitter::jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_exp_emitter::jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_exp_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_exp_emitter::aux_gprs_count() const {
    return 2;
}

size_t jit_exp_emitter::aux_vecs_count() const {
    return 3;
}

void jit_exp_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    VReg aux0 = VReg(aux_vec_idxs[0]);
    VReg aux1 = VReg(aux_vec_idxs[1]);
    VReg aux2 = VReg(aux_vec_idxs[2]);

    // save src
    h->vmv_v_v(aux2, src);

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    FReg ln_flt_min_f = f1;
    h->flw(ln_flt_min_f, p_table, 10 * sizeof(uint32_t));
    h->vfmax_vf(dst, src, ln_flt_min_f);

    FReg ln_flt_max_f = f0;
    h->flw(ln_flt_max_f, p_table, 9 * sizeof(uint32_t));
    h->vfmin_vf(dst, dst, ln_flt_max_f);

    // keep dst = x for further computations
    h->vmv_v_v(aux0, dst);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    FReg log2ef = f0;
    h->flw(log2ef, p_table, 8 * sizeof(uint32_t));
    h->vfmul_vf(dst, dst, log2ef);
    FReg half = f0;
    h->flw(half, p_table, 6 * sizeof(uint32_t));
    h->vfadd_vf(dst, dst, log2ef);

    // aux1 = floorf(fx)
    h->vfcvt_x_f_v(aux1, dst); // fp32 -> int32
    h->vfcvt_f_x_v(aux1, aux1); // int32 -> fp32
    h->vmfgt_vv(mask_vreg(), aux1, dst);
    FReg one = f0;
    h->flw(one, p_table, 5 * sizeof(uint32_t)); // one
    h->vfsub_vf(aux1, aux1, one, VM::masked);

    // keep dst = floorf(fx) for further computations
    h->vmv_v_v(dst, aux1);

    // x = x - fx * ln2
    FReg ln2 = f0;
    h->flw(ln2, p_table, 7 * sizeof(uint32_t));
    h->vfnmsac_vf(aux0, ln2, aux1);

    // compute 2^n
    Reg tmp = Reg(aux_gpr_idxs[0]);
    h->vfcvt_x_f_v(aux1, dst);
    h->lw(tmp, p_table, 11 * sizeof(uint32_t)); // exponent_bias
    h->vadd_vx(aux1, aux1, tmp);
    const int n_mantissa_bits = 23;
    h->vsll_vi(aux1, aux1, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    // Note: Xbyak doesn't support vmv_v_i with mask to set zero where masked
    h->vmflt_vf(mask_vreg(), aux2, ln_flt_min_f); // aux - tmp mask
    h->vand_vx(aux1, aux1, zero, VM::masked);

    // compute polynomial
    FReg pol = f0;
    h->flw(pol, p_table, 4 * sizeof(uint32_t)); // pol5
    h->vfmv_v_f(dst, pol);

    h->flw(pol, p_table, 3 * sizeof(uint32_t)); // pol4
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    h->flw(pol, p_table, 2 * sizeof(uint32_t)); // pol3
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    h->flw(pol, p_table, 1 * sizeof(uint32_t)); // pol2
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    h->flw(pol, p_table, 0 * sizeof(uint32_t)); // pol1
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    h->flw(pol, p_table, 5 * sizeof(uint32_t)); // one
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    // y = y * 2^n
    h->vfmul_vv(dst, dst, aux1);
}

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

bool jit_exp_emitter::need_table() const {
    return true;
}

const void* jit_exp_emitter::get_table() const {
    static uint32_t tbl[12] = {
        0x3f7ffffb, // pol1
        0x3efffee3, // pol2
        0x3e2aad40, // pol3
        0x3d2b9d0d, // pol4
        0x3c07cfce, // pol5
        0x3f800000, // one
        0x3f000000, // 0.5f
        0x3f317218, // ln2f
        0x3fb8aa3b, // log2ef
        0x42b17218, // ln_flt_max_f
        0xc2aeac50, // ln_flt_min_f
        0x0000007f  // exponent_bias
    };
    return tbl;
}

/// MUL ///
jit_mul_emitter::jit_mul_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_mul_emitter::jit_mul_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_mul_emitter::get_inputs_num() const {
    return 2;
}

void jit_mul_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfmul_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_mul_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// PReLU ///
jit_prelu_emitter::jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_prelu_emitter::jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_prelu_emitter::get_inputs_num() const {
    return 2;
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);
    FReg fzero = f0;

    if (src0.getIdx() != dst.getIdx())
        h->vmv_v_v(dst, src0);

    h->fmv_w_x(fzero, zero);
    h->vmflt_vf(mask_vreg(), src0, fzero);

    h->vfmul_vv(dst, src0, src1, VM::masked);
}

std::set<std::vector<element::Type>> jit_prelu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// ReLU ///
jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {
    if (const auto leaky_relu = ov::as_type_ptr<LeakyReluNode>(node)) {
        alpha = leaky_relu->get_slope();
    } else if (ov::is_type<ov::op::v0::Relu>(node)) {
        alpha = 0.f;
    } else {
        OPENVINO_THROW("Incompatible node!");
    }
}

jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, float alpha, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc), alpha(alpha) {}

size_t jit_relu_emitter::get_inputs_num() const {
    return 1;
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    FReg fzero = f0;

    h->fmv_w_x(fzero, zero);

    if (alpha == 0) {
        h->vfmax_vf(dst, src, fzero);
        return;
    }

    if (src.getIdx() != dst.getIdx())
        h->vmv_v_v(dst, src);

    h->vmflt_vf(mask_vreg(), dst, fzero);

    FReg alpha_reg = fzero;
    h->flw(alpha_reg, p_table);
    h->vfmul_vf(dst, dst, alpha_reg, VM::masked);
}

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

bool jit_relu_emitter::need_table() const {
    return alpha != 0;
}

const void* jit_relu_emitter::get_table() const {
    static float tbl[1];
    tbl[0] = alpha; // use explicit assignment to change dynamically array in runtime
    return tbl;
}

/// SUB ///
jit_sub_emitter::jit_sub_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_sub_emitter::jit_sub_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_sub_emitter::get_inputs_num() const {
    return 2;
}

void jit_sub_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfsub_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_sub_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov
