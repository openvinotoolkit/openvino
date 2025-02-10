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


/// CLamp ///
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
    static float values[2];
    values[0] = min; // use explicit assignment to change dynamically array in runtime
    values[1] = max;
    return values;
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
    static float values[1];
    values[0] = alpha; // use explicit assignment to change dynamically array in runtime
    return values;
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
