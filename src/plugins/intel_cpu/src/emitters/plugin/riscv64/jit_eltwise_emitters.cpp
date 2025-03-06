// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include "transformations/cpu_opset/common/op/leaky_relu.hpp"

namespace ov::intel_cpu::riscv64 {

using namespace Xbyak_riscv;

#define CONST_1_F    0x3f800000  // 1.f

/// ADD ///
jit_add_emitter::jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_add_emitter::jit_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_add_emitter::get_inputs_num() const {
    return 2;
}

void jit_add_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    switch (exec_prc_) {
    case ov::element::f32:
        h->vfadd_vv(dst, src0, src1);
        break;
    case ov::element::i32:
        h->vadd_vv(dst, src0, src1);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
    }
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// Clamp ///
jit_clamp_emitter::jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    if (const auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(node)) {
        min = static_cast<float>(clamp->get_min());
        max = static_cast<float>(clamp->get_max());
    } else {
        OPENVINO_THROW("Incompatible node!");
    }
    prepare_table();
}

jit_clamp_emitter::jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     float min, float max, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc), min(min), max(max) {
    prepare_table();
}

size_t jit_clamp_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_clamp_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_clamp_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_clamp_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    FReg bound = FReg(aux_fp_gpr_idxs[0]);

    load_table_val("min", bound);
    h->vfmax_vf(dst, src, bound);

    load_table_val("max", bound);
    h->vfmin_vf(dst, dst, bound);
}

void jit_clamp_emitter::register_table_entries() {
    push_arg_entry_of("min", dnnl::impl::float2int(min));
    push_arg_entry_of("max", dnnl::impl::float2int(max));
}

std::set<std::vector<element::Type>> jit_clamp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// DIV ///
jit_divide_emitter::jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_divide_emitter::jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator* host,  ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_num() const {
    return 2;
}

void jit_divide_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_divide_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfdiv_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// Exp ///
jit_exp_emitter::jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_exp_emitter::jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_exp_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_exp_emitter::aux_gprs_count() const {
    return 2;
}

size_t jit_exp_emitter::aux_vecs_count() const {
    return 3;
}

size_t jit_exp_emitter::aux_fp_gprs_count() const {
    return 2;
}

void jit_exp_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_exp_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    VReg aux0 = VReg(aux_vec_idxs[0]);
    VReg aux1 = VReg(aux_vec_idxs[1]);
    VReg zero_mask = VReg(aux_vec_idxs[2]);
    VReg aux2 = zero_mask;
    FReg fp0 = FReg(aux_fp_gpr_idxs[0]);
    FReg fp1 = FReg(aux_fp_gpr_idxs[1]);
    Reg tmp = Reg(aux_gpr_idxs[0]);

    FReg ln_flt_min_f = fp0;
    load_table_val("ln_flt_min_f", ln_flt_min_f);
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->vmflt_vf(mask_vreg(), src, ln_flt_min_f);
    h->vmv1r_v(zero_mask, mask_vreg()); // save mask

    h->vfmax_vf(dst, src, ln_flt_min_f);

    load_table_val("ln_flt_max_f", fp1);
    h->vfmin_vf(dst, dst, fp1);

    // keep dst = x for further computations
    h->vmv_v_v(aux0, dst);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    load_table_val("log2ef", fp1);
    h->vfmul_vf(dst, dst, fp1);
    load_table_val("half", fp1);
    h->vfadd_vf(dst, dst, fp1);

    // aux1 = floorf(fx)
    h->vfcvt_x_f_v(aux1, dst); // fp32 -> int32
    h->vfcvt_f_x_v(aux1, aux1); // int32 -> fp32
    h->vmfgt_vv(mask_vreg(), aux1, dst);
    load_table_val("one", fp1);
    h->vfsub_vf(aux1, aux1, fp1, VM::masked);

    // keep dst = floorf(fx) for further computations
    h->vmv_v_v(dst, aux1);

    // x = x - fx * ln2
    load_table_val("ln2f", fp1);
    h->vfnmsac_vf(aux0, fp1, aux1);

    // compute 2^n
    h->vfcvt_x_f_v(aux1, dst);
    load_table_val("exponent_bias", tmp);
    h->vadd_vx(aux1, aux1, tmp);
    const int n_mantissa_bits = 23;
    h->vsll_vi(aux1, aux1, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    // Note: Xbyak doesn't support vmv_v_i with mask to set zero where masked
    h->vmv1r_v(mask_vreg(), zero_mask); // pop mask
    h->vand_vx(aux1, aux1, zero, VM::masked);

    // compute polynomial
    FReg pol = fp1;
    load_table_val("pol5", pol);
    h->vfmv_v_f(dst, pol);

    load_table_val("pol4", pol);
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    load_table_val("pol3", pol);
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    load_table_val("pol2", pol);
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    load_table_val("pol1", pol);
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    load_table_val("one", pol);
    h->vfmv_v_f(aux2, pol);
    h->vfmadd_vv(dst, aux0, aux2);

    // y = y * 2^n
    h->vfmul_vv(dst, dst, aux1);
}

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_exp_emitter::register_table_entries() {
    push_arg_entry_of("pol1", 0x3f7ffffb);  // p1 = 0.999999701f
    push_arg_entry_of("pol2", 0x3efffee3);  // p2 = 0.499991506f
    push_arg_entry_of("pol3", 0x3e2aad40);  // p3 = 0.166676521f
    push_arg_entry_of("pol4", 0x3d2b9d0d);  // p4 = 0.0418978221f
    push_arg_entry_of("pol5", 0x3c07cfce);  // p5 = 0.00828929059f

    push_arg_entry_of("one", CONST_1_F);
    push_arg_entry_of("half", 0x3f000000);
    push_arg_entry_of("ln2f", 0x3f317218);
    push_arg_entry_of("log2ef", 0x3fb8aa3b);
    push_arg_entry_of("ln_flt_max_f", 0x42b17218);
    push_arg_entry_of("ln_flt_min_f", 0xc2aeac50);
    push_arg_entry_of("exponent_bias", 0x0000007f);
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
    const std::shared_ptr<ov::Node>& node)
: jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_mul_add_emitter::jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
    const ov::element::Type exec_prc)
: jit_emitter(host, host_isa, exec_prc) {}

size_t jit_mul_add_emitter::get_inputs_num() const {
    return 3;
}

void jit_mul_add_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg src2 = VReg(in_vec_idxs[2]);
    VReg dst = VReg(out_vec_idxs[0]);

    if (src1.getIdx() == dst.getIdx()) {
        h->vfmadd_vv(dst, src0, src2);
        return;
    }

    if (src2.getIdx() == dst.getIdx()) {
        h->vfmacc_vv(dst, src0, src1);
        return;
    }

    if (src0.getIdx() != dst.getIdx()) {
        h->vmv_v_v(dst, src0);
    }
    h->vfmadd_vv(dst, src1, src2);
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

/// MUL ///
jit_multiply_emitter::jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_num() const {
    return 2;
}

void jit_multiply_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    h->vfmul_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// PReLU ///
jit_prelu_emitter::jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_prelu_emitter::jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_prelu_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_prelu_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_prelu_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);
    FReg fzero = FReg(aux_fp_gpr_idxs[0]);

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
jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    if (const auto leaky_relu = ov::as_type_ptr<LeakyReluNode>(node)) {
        alpha = leaky_relu->get_slope();
    } else if (ov::is_type<ov::op::v0::Relu>(node)) {
        alpha = 0.f;
    } else {
        OPENVINO_THROW("Incompatible node!");
    }
    prepare_table();
}

jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   float alpha, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc), alpha(alpha) {
    prepare_table();
}

size_t jit_relu_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_relu_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_relu_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    FReg fzero = FReg(aux_fp_gpr_idxs[0]);
    h->fmv_w_x(fzero, zero);

    if (alpha == 0) {
        h->vfmax_vf(dst, src, fzero);
        return;
    }

    if (src.getIdx() != dst.getIdx())
        h->vmv_v_v(dst, src);

    h->vmflt_vf(mask_vreg(), dst, fzero);

    FReg alpha_reg = fzero;
    load_table_val("alpha", alpha_reg);
    h->vfmul_vf(dst, dst, alpha_reg, VM::masked);
}

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_relu_emitter::register_table_entries() {
    if (alpha != 0)
        push_arg_entry_of("alpha", dnnl::impl::float2int(alpha));
}

/// Power Static ///
jit_power_static_emitter::jit_power_static_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                   float power, float scale, float shift, ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc), power(power), scale(scale), shift(shift) {
    prepare_table();
}

size_t jit_power_static_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_power_static_emitter::aux_gprs_count() const {
    if ((power == 0) || is_scale_shift() || (!is_sqrt() && !is_int_pow()))
        return 2;
    return 1;
}

bool jit_power_static_emitter::is_lmul_supported() const {
    return jit_emitter::is_lmul_supported() && (is_int_pow() || is_sqrt());
}

size_t jit_power_static_emitter::aux_vecs_count() const {
    if (is_scale_shift())
        return 2;
    if (is_int_pow())
        return 1;
    return 0;
}

size_t jit_power_static_emitter::aux_fp_gprs_count() const {
    return power < 0 ? 1 : 0;
}

void jit_power_static_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_power_static_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);

    if (power == 0) {
        Reg tmp = Reg(aux_gpr_idxs[0]);
        load_table_val("one", dst, tmp);
        return;
    }

    if (is_scale_shift()) {
        VReg aux0 = VReg(aux_vec_idxs[0]);
        VReg aux1 = VReg(aux_vec_idxs[1]);
        Reg tmp = Reg(aux_gpr_idxs[0]);
        load_table_val("shift", aux0, tmp);
        load_table_val("scale", aux1, tmp);
        h->vfmacc_vv(aux0, aux1, src);
        h->vmv_v_v(dst, aux0);
    } else {
        if (src.getIdx() != dst.getIdx())
            h->vmv_v_v(dst, src);
    }

    // for power `-0.5f` there is `vfrsqrt7_v` instruction with worse accuracy
    if (is_sqrt()) {
        h->vfsqrt_v(dst, dst);

        if (power < 0) {
            FReg one = FReg(aux_fp_gpr_idxs[0]);
            load_table_val("one", one);
            h->vfrdiv_vf(dst, dst, one);
        }
    } else if (is_int_pow()) {
        int64_t ipower = std::abs(static_cast<int64_t>(power)) - 1;

        VReg aux0 = VReg(aux_vec_idxs[0]);
        h->vmv_v_v(aux0, dst);

        while (ipower > 0) {
            if (ipower & 0x1) {
                h->vfmul_vv(dst, dst, aux0);
            }
            if (ipower > 1) {
                h->vfmul_vv(aux0, aux0, aux0);
            }
            ipower = ipower >> 1;
        }

        if (power < 0) {
            FReg one = FReg(aux_fp_gpr_idxs[0]);
            load_table_val("one", one);
            h->vfrdiv_vf(dst, dst, one);
        }
    } else {
        auto pow_f32_addr = reinterpret_cast<uintptr_t>(::powf);

        Reg func_reg(aux_gpr_idxs[0]);
        h->uni_li(func_reg, pow_f32_addr);

        // Before binary call we have to save caller-saver registers:
        // - all caller-saver general-purpose regs + func_reg (if it's caller-saver)
        // - all caller-saver fp general-purpose regs except aux registers
        // - all vector registers except aux, src and dst registers
        auto exclude_vec_regs = aux_vec_idxs;
        aux_vec_idxs.push_back(src.getIdx());
        aux_vec_idxs.push_back(dst.getIdx());
        call_preamble({}, aux_fp_gpr_idxs, aux_vec_idxs);

        const auto sp_size = rnd_up(get_vec_length(), 16);
        h->addi(sp, sp, -sp_size);
        h->vse32_v(dst, sp);

        // TODO: Support any LMUL here (via vl from csr + labels)
        for (size_t i = 0; i < get_vec_length(); i += sizeof(float)) {
            h->flw(fa0, sp, i);
            load_table_val("power", fa1);

            h->jalr(ra, func_reg);

            h->fsw(fa0, sp, i);
        }

        h->vle32_v(dst, sp);
        h->addi(sp, sp, sp_size);

        call_postamble({}, aux_fp_gpr_idxs, aux_vec_idxs);
    }
}

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_power_static_emitter::register_table_entries() {
    if (scale != 1.f || shift != 0.f) {
        push_arg_entry_of("scale", dnnl::impl::float2int(scale));
        push_arg_entry_of("shift", dnnl::impl::float2int(shift));
    }
    if (power != 1.f)
        push_arg_entry_of("power", dnnl::impl::float2int(power));
    if (power < 0)
        push_arg_entry_of("one", CONST_1_F);
}

/// Sigmoid ///
jit_sigmoid_emitter::jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node, const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    jit_exp_emitter_.reset(new jit_exp_emitter(host, host_isa, exec_prc));
    prepare_table();
}

jit_sigmoid_emitter::jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    jit_exp_emitter_.reset(new jit_exp_emitter(host, host_isa, exec_prc));
    prepare_table();
}

size_t jit_sigmoid_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_sigmoid_emitter::aux_gprs_count() const {
    OPENVINO_ASSERT(jit_exp_emitter_, "JIT Exp emitter is missed!");
    return jit_exp_emitter_->aux_gprs_count();
}

size_t jit_sigmoid_emitter::aux_vecs_count() const {
    OPENVINO_ASSERT(jit_exp_emitter_, "JIT Exp emitter is missed!");
    return jit_exp_emitter_->aux_vecs_count() + 1;
}

size_t jit_sigmoid_emitter::aux_fp_gprs_count() const {
    OPENVINO_ASSERT(jit_exp_emitter_, "JIT Exp emitter is missed!");
    return std::max(jit_exp_emitter_->aux_fp_gprs_count(), 1lu);
}

void jit_sigmoid_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_sigmoid_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    VReg sign_mask = VReg(aux_vec_idxs[aux_vecs_count() - 1]);
    VReg aux = VReg(aux_vec_idxs[aux_vecs_count() - 2]);

    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.

    // we store the original sign and make x negative
    FReg fzero = FReg(aux_fp_gpr_idxs[0]);
    h->vmfgt_vf(mask_vreg(), src, fzero);
    h->vfneg_vv(src, src, VM::masked);
    h->vmv1r_v(sign_mask, mask_vreg()); // save mask since exp uses mask too

    const auto exp_src_idxs = std::vector<size_t>{static_cast<size_t>(src.getIdx())};
    const auto exp_dst_idxs = std::vector<size_t>{static_cast<size_t>(dst.getIdx())};
    const auto exp_aux_vec_idxs = std::vector<size_t>{aux_vec_idxs.cbegin(), aux_vec_idxs.cbegin() + jit_exp_emitter_->aux_vecs_count()};
    jit_exp_emitter_->emit_code(exp_src_idxs, exp_dst_idxs, exp_aux_vec_idxs, aux_gpr_idxs, aux_fp_gpr_idxs);

    FReg one = FReg(aux_fp_gpr_idxs[0]);
    load_table_val("one", one);
    // aux = copy exp(x)
    h->vmv_v_v(aux, dst);
    // aux = (exp(x) + 1)
    h->vfadd_vf(aux, aux, one);
    // dst = exp(x) / (exp(x) + 1) = dst / aux
    h->vfdiv_vv(dst, dst, aux);

    // Now we have to apply the "symmetry" based on original sign
    // aux = dst - 1 = 1 - ( 1 / (exp(x) + 1))
    h->vfrsub_vf(aux, dst, one);
    h->vmv1r_v(mask_vreg(), sign_mask); // pop mask
    h->vmerge_vvm(dst, dst, aux);
}

void jit_sigmoid_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

void jit_sigmoid_emitter::emit_data() const {
    jit_emitter::emit_data();
    jit_exp_emitter_->emit_data();
}

std::set<std::vector<element::Type>> jit_sigmoid_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// SUB ///
jit_subtract_emitter::jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_subtract_emitter::jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_subtract_emitter::get_inputs_num() const {
    return 2;
}

void jit_subtract_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src0 = VReg(in_vec_idxs[0]);
    VReg src1 = VReg(in_vec_idxs[1]);
    VReg dst = VReg(out_vec_idxs[0]);

    switch (exec_prc_) {
    case ov::element::f32:
        h->vfsub_vv(dst, src0, src1);
        break;
    case ov::element::i32:
        h->vsub_vv(dst, src0, src1);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision");
    }
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

#undef CONST_1_F

}  // ov::intel_cpu::riscv64
