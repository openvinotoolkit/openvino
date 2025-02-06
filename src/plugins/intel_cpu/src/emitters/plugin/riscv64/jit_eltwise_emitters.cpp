// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

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

/// ReLU ///
jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, get_arithmetic_binary_exec_precision(node)) {}

jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator* host, const ov::element::Type exec_prc)
    : jit_emitter(host, exec_prc) {}

size_t jit_relu_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_relu_emitter::aux_vecs_count() const {
    return 1;
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    VReg src = VReg(in_vec_idxs[0]);
    VReg dst = VReg(out_vec_idxs[0]);
    VReg zeros = VReg(aux_vec_idxs[0]);

    h->vmv_v_i(zeros, 0);
    h->vfmax_vv(dst, src, zeros);
}

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
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
