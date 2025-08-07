// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

#include "common/utils.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/utils.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/relu.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

using namespace Xbyak_riscv;

#define CONST_1_F 0x3f800000  // 1.F

/// ABS ///
jit_abs_emitter::jit_abs_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_abs_emitter::jit_abs_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_abs_emitter::get_inputs_num() const {
    return 1;
}

void jit_abs_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_abs_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    h->vfsgnjx_vv(dst, src, src);
}

std::set<std::vector<element::Type>> jit_abs_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// ADD ///
jit_add_emitter::jit_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_add_emitter::jit_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_add_emitter::get_inputs_num() const {
    return 2;
}

void jit_add_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    switch (exec_prc_) {
    case ov::element::f32:
        h->vfadd_vv(dst, src0, src1);
        break;
    case ov::element::i32:
        h->vadd_vv(dst, src0, src1);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", exec_prc_);
    }
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// Clamp ///
jit_clamp_emitter::jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node,
                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    if (const auto clamp = ov::as_type_ptr<ov::op::v0::Clamp>(node)) {
        min = static_cast<float>(clamp->get_min());
        max = static_cast<float>(clamp->get_max());
    } else {
        OV_CPU_JIT_EMITTER_THROW("Incompatible node!");
    }
    prepare_table();
}

jit_clamp_emitter::jit_clamp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     float min,
                                     float max,
                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      min(min),
      max(max) {
    prepare_table();
}

size_t jit_clamp_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_clamp_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_clamp_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_clamp_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);
    auto bound = FReg(aux_fp_gpr_idxs[0]);

    load_table_val("min", bound);
    h->vfmax_vf(dst, src, bound);

    load_table_val("max", bound);
    h->vfmin_vf(dst, dst, bound);
}

void jit_clamp_emitter::register_table_entries() {
    push_arg_entry_of("min", dnnl::impl::float2int(min));
    push_arg_entry_of("max", dnnl::impl::float2int(max));
}

std::set<std::vector<element::Type>> jit_clamp_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// DIV ///
jit_divide_emitter::jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_divide_emitter::jit_divide_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_num() const {
    return 2;
}

void jit_divide_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_divide_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    switch (exec_prc_) {
    case ov::element::f32:
        h->vfdiv_vv(dst, src0, src1);
        break;
    case ov::element::i32:
        h->vdiv_vv(dst, src0, src1);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", exec_prc_);
    }
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

/// Elu ///
jit_elu_emitter::jit_elu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 float alpha,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      alpha(alpha) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

jit_elu_emitter::jit_elu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    const auto elu = ov::as_type_ptr<ov::op::v0::Elu>(node);
    if (elu == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to ov::op::v0::Elu");
    }
    alpha = static_cast<float>(elu->get_alpha());
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_elu_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_elu_emitter::aux_gprs_count() const {
    return exp_emitter->aux_gprs_count() + 1;
}

size_t jit_elu_emitter::aux_vecs_count() const {
    return std::max<size_t>(exp_emitter->aux_vecs_count() + 1, 2);
}

size_t jit_elu_emitter::aux_fp_gprs_count() const {
    return std::max<size_t>(exp_emitter->aux_fp_gprs_count(), 1);
}

void jit_elu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_elu_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    auto aux0 = VReg(aux_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[1]);
    auto fp0 = FReg(aux_fp_gpr_idxs[0]);

    h->vmv_v_v(aux1, src);

    // compute exponent
    auto exp_aux_vec_idxs = aux_vec_idxs;
    exp_aux_vec_idxs.erase(
        std::find(exp_aux_vec_idxs.begin(), exp_aux_vec_idxs.end(), static_cast<size_t>(aux1.getIdx())));
    exp_emitter->emit_code({static_cast<size_t>(src.getIdx())},
                           {static_cast<size_t>(dst.getIdx())},
                           exp_aux_vec_idxs,
                           aux_gpr_idxs,
                           aux_fp_gpr_idxs);

    load_table_val("one", fp0);
    h->vfsub_vf(dst, dst, fp0);  // dst = exp(x)-1
    load_table_val("alpha", fp0);
    h->vfmul_vf(dst, dst, fp0);  // dst = alpha * (exp(x) - 1)

    h->vmv_v_x(aux0, zero);
    h->vmfgt_vv(mask_vreg(), aux1, aux0);
    h->vmerge_vvm(dst, dst, aux1);
}

std::set<std::vector<element::Type>> jit_elu_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_elu_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
    push_arg_entry_of("alpha", dnnl::impl::float2int(alpha));
}

void jit_elu_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

/// EQUAL ///
jit_equal_emitter::jit_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_equal_emitter::jit_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_equal_emitter::get_inputs_num() const {
    return 2;
}
size_t jit_equal_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

std::set<std::vector<element::Type>> jit_equal_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_equal_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);
    auto one = FReg(aux_fp_gpr_idxs[0]);
    load_table_val("one", one);

    h->vmfeq_vv(mask_vreg(), src0, src1);    // compare, result in mask
    h->vmv_v_x(dst, zero);                   // set dst to 0
    h->vfadd_vf(dst, dst, one, VM::masked);  // set 1.0 where mask is true
}

/// Erf ///
jit_erf_emitter::jit_erf_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

jit_erf_emitter::jit_erf_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_erf_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_erf_emitter::aux_gprs_count() const {
    return std::max<size_t>(exp_emitter->aux_gprs_count(), 1) + 1;
}

size_t jit_erf_emitter::aux_vecs_count() const {
    return std::max<size_t>(exp_emitter->aux_vecs_count() + 1, 4);
}

size_t jit_erf_emitter::aux_fp_gprs_count() const {
    return std::max<size_t>(exp_emitter->aux_fp_gprs_count(), 1);
}

void jit_erf_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_erf_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    auto aux0 = VReg(aux_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[1]);
    auto aux2 = VReg(aux_vec_idxs[2]);
    auto aux3 = VReg(aux_vec_idxs[3]);
    auto fp0 = FReg(aux_fp_gpr_idxs[0]);
    auto tmp = Reg(aux_gpr_idxs[0]);

    // store x in aux3
    h->vmv_v_v(aux3, src);

    // -exp(-x*x)
    h->vfmul_vv(dst, src, src);
    h->vfneg_v(dst, dst);

    // remove the already used aux3
    auto exp_aux_vec_idxs = aux_vec_idxs;
    exp_aux_vec_idxs.erase(
        std::find(exp_aux_vec_idxs.begin(), exp_aux_vec_idxs.end(), static_cast<size_t>(aux3.getIdx())));
    exp_emitter->emit_code({static_cast<size_t>(dst.getIdx())},
                           {static_cast<size_t>(dst.getIdx())},
                           exp_aux_vec_idxs,
                           aux_gpr_idxs,
                           aux_fp_gpr_idxs);
    h->vfneg_v(dst, dst);

    // save sign in aux0
    h->vmv_v_v(aux0, aux3);

    // aux1 = abs(x)
    h->vfabs_v(aux1, aux3);

    // t = 1 / (p*|x| + 1)
    load_table_val("erf_approx_const", fp0);
    h->vfmul_vf(aux2, aux1, fp0);  // aux2 = p * |x|
    load_table_val("one", fp0);
    h->vfadd_vf(aux2, aux2, fp0);  // aux2 = p * |x| + 1
    h->vfmv_v_f(aux1, fp0);
    h->vfdiv_vv(aux1, aux1, aux2);  // aux1 = 1 / (p * |x| + 1)

    // dst = - t * exp(-x*x)
    h->vfmul_vv(dst, dst, aux1);

    // compute polynomial r
    load_table_val("erf_pol4", aux3, tmp);  // aux3 = p4
    load_table_val("erf_pol5", fp0);
    h->vfmacc_vf(aux3, fp0, aux1);  // aux3 = p5 * t + p4

    load_table_val("erf_pol3", aux2, tmp);  // aux2 = p3
    h->vfmacc_vv(aux2, aux3, aux1);         // aux2 = p5 * t^2 + p4 * t + p3

    load_table_val("erf_pol2", aux3, tmp);  // aux3 = p2
    h->vfmacc_vv(aux3, aux2, aux1);         // aux3 = p5 * t^3 + p4 * t^2 + p3 * t + p2

    load_table_val("erf_pol1", aux2, tmp);  // aux2 = p1
    h->vfmacc_vv(aux2, aux3, aux1);         // aux2 = p5 * t^4 + p4 * t^3 + p3 * t^2 + p2 * t + p1

    // erf = sign * (1 - r * t * exp(-x * x))
    load_table_val("one", aux3, tmp);  // aux3 = 1
    h->vfmacc_vv(aux3, aux2, dst);     // aux3 = 1 - r * t * exp(-x * x)
    // add signs back
    h->vfsgnj_vv(dst, aux3, aux0);
}

std::set<std::vector<element::Type>> jit_erf_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_erf_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
    push_arg_entry_of("erf_approx_const", 0x3ea7ba05);  // 0.3275911

    push_arg_entry_of("erf_pol1", 0x3e827906);  // p1 = 0.254829592f
    push_arg_entry_of("erf_pol2", 0xbe91a98e);  // p2 = -0.284496736f
    push_arg_entry_of("erf_pol3", 0x3fb5f0e3);  // p3 = 1.421413741f
    push_arg_entry_of("erf_pol4", 0xbfba00e3);  // p4 = -1.453152027f
    push_arg_entry_of("erf_pol5", 0x3f87dc22);  // p5 = 1.061405429f
}

void jit_erf_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

/// Exp ///
jit_exp_emitter::jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_exp_emitter::jit_exp_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 ov::element::Type exec_prc)
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
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_exp_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);
    auto aux0 = VReg(aux_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[1]);
    auto zero_mask = VReg(aux_vec_idxs[2]);
    VReg aux2 = zero_mask;
    auto fp0 = FReg(aux_fp_gpr_idxs[0]);
    auto fp1 = FReg(aux_fp_gpr_idxs[1]);
    auto tmp = Reg(aux_gpr_idxs[0]);

    FReg ln_flt_min_f = fp0;
    load_table_val("ln_flt_min_f", ln_flt_min_f);
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->vmflt_vf(mask_vreg(), src, ln_flt_min_f);
    h->vmv1r_v(zero_mask, mask_vreg());  // save mask

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
    h->vfcvt_x_f_v(aux1, dst);   // fp32 -> int32
    h->vfcvt_f_x_v(aux1, aux1);  // int32 -> fp32
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
    h->vmv1r_v(mask_vreg(), zero_mask);  // pop mask
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

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
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

/// MOD ///
jit_mod_emitter::jit_mod_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_mod_emitter::jit_mod_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_mod_emitter::get_inputs_num() const {
    return 2;
}
size_t jit_mod_emitter::aux_vecs_count() const {
    if (exec_prc_ == ov::element::f32) {
        return 2;
    }
    if (exec_prc_ == ov::element::i32) {
        return 0;
    }
    OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", exec_prc_);
}
size_t jit_mod_emitter::aux_fp_gprs_count() const {
    if (exec_prc_ == ov::element::f32) {
        return 1;
    }
    if (exec_prc_ == ov::element::i32) {
        return 0;
    }
    OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", exec_prc_);
}
void jit_mod_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_mod_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    switch (exec_prc_) {
    case ov::element::i32:
        h->vremu_vv(dst, src0, src1);
        break;
    case ov::element::f32: {
        auto tmp0 = VReg(aux_vec_idxs[0]);
        auto tmp1 = VReg(aux_vec_idxs[1]);
        auto fp0 = FReg(aux_fp_gpr_idxs[0]);
        h->vfdiv_vv(tmp0, src0, src1);
        h->vfcvt_x_f_v(tmp1, tmp0);
        h->vfcvt_f_x_v(tmp1, tmp1);
        h->vmfgt_vv(mask_vreg(), tmp1, tmp0);
        load_table_val("one", fp0);
        h->vfsub_vf(tmp1, tmp1, fp0, VM::masked);
        h->vfmul_vv(tmp0, tmp1, src1);
        h->vfsub_vv(dst, src0, tmp0);
        break;
    }
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision:", exec_prc_);
    }
}
std::set<std::vector<element::Type>> jit_mod_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::i32, element::i32}, {element::f32, element::f32}};
}
void jit_mod_emitter::register_table_entries() {
    if (exec_prc_ == ov::element::f32) {
        push_arg_entry_of("one", CONST_1_F);
    }
}

/// FLOOR ///
jit_floor_emitter::jit_floor_emitter(jit_generator_t* host, cpu_isa_t host_isa, const element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_floor_emitter::jit_floor_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

size_t jit_floor_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_floor_emitter::aux_vecs_count() const {
    return 1;
}

size_t jit_floor_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_floor_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for FLOOR");
    }
}
void jit_floor_emitter::register_table_entries() {
    push_arg_entry_of("neg_one", 0xbf800000);
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_floor_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[0]);
    auto fp1 = FReg(aux_fp_gpr_idxs[0]);

    h->vmv_v_v(aux1, src);
    h->vfcvt_x_f_v(dst, src);
    h->vfcvt_f_x_v(dst, dst);

    h->vmfgt_vv(mask_vreg(), dst, aux1);
    load_table_val("neg_one", fp1);
    h->vfadd_vf(dst, dst, fp1, VM::masked);
}
std::set<std::vector<element::Type>> jit_floor_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}
/// FLOOR MOD ///
jit_floor_mod_emitter::jit_floor_mod_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_floor_mod_emitter::jit_floor_mod_emitter(jit_generator_t* host,
                                             cpu_isa_t host_isa,
                                             const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

size_t jit_floor_mod_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_floor_mod_emitter::aux_vecs_count() const {
    return 2;
}

size_t jit_floor_mod_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_floor_mod_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                      const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for FLOOR_MOD");
    }
}
template <cpu_isa_t isa>
void jit_floor_mod_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == element::f32, "JIT Floor Mod emitter supports only f32 precision");

    const VReg src0 = VReg(in_vec_idxs[0]);
    const VReg src1 = VReg(in_vec_idxs[1]);
    const VReg dst = VReg(out_vec_idxs[0]);
    const VReg tmp1 = VReg(aux_vec_idxs[0]);
    const VReg tmp2 = VReg(aux_vec_idxs[1]);
    auto fone = FReg(aux_fp_gpr_idxs[0]);

    load_table_val("one", fone);

    h->vfdiv_vv(tmp1, src0, src1);

    h->vfcvt_x_f_v(tmp2, tmp1);
    h->vfcvt_f_x_v(tmp2, tmp2);
    h->vmflt_vv(mask_vreg(), tmp1, tmp2);
    h->vfsub_vf(tmp2, tmp2, fone, VM::masked);
    h->vfmul_vv(tmp1, tmp2, src1);
    h->vfsub_vv(dst, src0, tmp1);
}
void jit_floor_mod_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}
std::set<std::vector<element::Type>> jit_floor_mod_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}
/// GELU ERF ///
jit_gelu_erf_emitter::jit_gelu_erf_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    erf_emitter = std::make_unique<jit_erf_emitter>(h, host_isa, exec_prc);
}

jit_gelu_erf_emitter::jit_gelu_erf_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    erf_emitter = std::make_unique<jit_erf_emitter>(h, host_isa, exec_prc);
}

size_t jit_gelu_erf_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_gelu_erf_emitter::aux_vecs_count() const {
    return erf_emitter->aux_vecs_count() + 1;
}

size_t jit_gelu_erf_emitter::aux_gprs_count() const {
    return erf_emitter->aux_gprs_count() + 1;
}

size_t jit_gelu_erf_emitter::aux_fp_gprs_count() const {
    return std::max(erf_emitter->aux_fp_gprs_count(), 1LU);
}

void jit_gelu_erf_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for GELU ERF");
    }
}

void jit_gelu_erf_emitter::register_table_entries() {
    push_arg_entry_of("one", 0x3f800000);
    push_arg_entry_of("half", 0x3f000000);
    push_arg_entry_of("one_over_sqrt_two", 0x3f3504f3);
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_gelu_erf_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    auto aux0 = VReg(aux_vec_idxs[erf_emitter->aux_vecs_count()]);

    auto fp0 = FReg(aux_fp_gpr_idxs[0]);

    // x = src / sqrt(2)
    load_table_val("one_over_sqrt_two", fp0);
    h->vfmul_vf(aux0, src, fp0);

    // erf(x)
    erf_emitter->emit_code({static_cast<size_t>(aux0.getIdx())},
                           {static_cast<size_t>(aux0.getIdx())},
                           {aux_vec_idxs.begin(), aux_vec_idxs.begin() + erf_emitter->aux_vecs_count()},
                           aux_gpr_idxs,
                           aux_fp_gpr_idxs);

    // 1 + erf(x)
    load_table_val("one", fp0);
    h->vfadd_vf(aux0, aux0, fp0);

    // 0.5 * (1 + erf(x))
    load_table_val("half", fp0);
    h->vfmul_vf(aux0, aux0, fp0);

    // x * 0.5 * (1 + erf(x))
    h->vfmul_vv(dst, aux0, src);
}

std::set<std::vector<element::Type>> jit_gelu_erf_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_gelu_erf_emitter::emit_data() const {
    erf_emitter->emit_data();
    jit_emitter::emit_data();
}

/// GREATER EQUAL ///
jit_greater_equal_emitter::jit_greater_equal_emitter(jit_generator_t* host,
                                                     cpu_isa_t host_isa,
                                                     const element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_greater_equal_emitter::jit_greater_equal_emitter(jit_generator_t* host,
                                                     cpu_isa_t host_isa,
                                                     const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

size_t jit_greater_equal_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_greater_equal_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_greater_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                          const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for GREATER_EQUAL");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_greater_equal_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                         const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    auto one = FReg(aux_fp_gpr_idxs[0]);
    load_table_val("one", one);

    h->vmfge_vv(mask_vreg(), src0, src1);
    h->vmv_v_x(dst, zero);
    h->vfadd_vf(dst, dst, one, VM::masked);
}

void jit_greater_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

std::set<std::vector<element::Type>> jit_greater_equal_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// HSIGMOID ///
jit_hsigmoid_emitter::jit_hsigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_hsigmoid_emitter::jit_hsigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_hsigmoid_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_hsigmoid_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_hsigmoid_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_hsigmoid_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    auto fp0 = FReg(aux_fp_gpr_idxs[0]);

    // result = (min(max(x + 3, 0), 6)) / 6
    load_table_val("three", fp0);
    h->vfadd_vf(dst, src, fp0);

    h->fmv_w_x(fp0, zero);
    h->vfmax_vf(dst, dst, fp0);

    load_table_val("six", fp0);
    h->vfmin_vf(dst, dst, fp0);
    load_table_val("one_sixth", fp0);
    h->vfmul_vf(dst, dst, fp0);
}

std::set<std::vector<element::Type>> jit_hsigmoid_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_hsigmoid_emitter::register_table_entries() {
    push_arg_entry_of("three", 0x40400000);
    push_arg_entry_of("six", 0x40c00000);
    push_arg_entry_of("one_sixth", dnnl::impl::float2int(1.F / 6.F));
}

/// HSWISH ///
jit_hswish_emitter::jit_hswish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    hsigmoid_emitter = std::make_unique<jit_hsigmoid_emitter>(host, host_isa, exec_prc);
}

jit_hswish_emitter::jit_hswish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                       [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                       ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    hsigmoid_emitter = std::make_unique<jit_hsigmoid_emitter>(host, host_isa, exec_prc);
}

size_t jit_hswish_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_hswish_emitter::aux_gprs_count() const {
    return hsigmoid_emitter->aux_gprs_count();
}

size_t jit_hswish_emitter::aux_vecs_count() const {
    return hsigmoid_emitter->aux_vecs_count() + 1;
}

size_t jit_hswish_emitter::aux_fp_gprs_count() const {
    return hsigmoid_emitter->aux_fp_gprs_count();
}

void jit_hswish_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_hswish_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    auto aux0 = VReg(aux_vec_idxs.back());

    // save src
    h->vmv_v_v(aux0, src);

    hsigmoid_emitter->emit_code({static_cast<size_t>(src.getIdx())},
                                {static_cast<size_t>(dst.getIdx())},
                                {aux_vec_idxs.begin(), aux_vec_idxs.begin() + hsigmoid_emitter->aux_vecs_count()},
                                aux_gpr_idxs,
                                aux_fp_gpr_idxs);
    h->vfmul_vv(dst, dst, aux0);
}

std::set<std::vector<element::Type>> jit_hswish_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_hswish_emitter::emit_data() const {
    hsigmoid_emitter->emit_data();
    jit_emitter::emit_data();
}
// LESS ///
jit_less_emitter::jit_less_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_less_emitter::jit_less_emitter(jit_generator_t* host, cpu_isa_t host_isa, const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}
size_t jit_less_emitter::get_inputs_num() const {
    return 2;
}
size_t jit_less_emitter::aux_fp_gprs_count() const {
    return 1;
}
void jit_less_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for LESS");
    }
}
template <cpu_isa_t isa>
void jit_less_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == element::f32, "JIT Less emitter supports only f32 precision");
    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);
    auto one = FReg(aux_fp_gpr_idxs[0]);
    load_table_val("one", one);

    h->vmflt_vv(mask_vreg(), src0, src1);
    h->vmv_v_x(dst, x0);
    h->vfadd_vf(dst, dst, one, VM::masked);
}
void jit_less_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}
std::set<std::vector<element::Type>> jit_less_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}
/// LOGICAL OR ///
jit_logical_or_emitter::jit_logical_or_emitter(jit_generator_t* host, cpu_isa_t host_isa, element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}
jit_logical_or_emitter::jit_logical_or_emitter(jit_generator_t* host,
                                               cpu_isa_t host_isa,
                                               const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}
size_t jit_logical_or_emitter::get_inputs_num() const {
    return 2;
}
size_t jit_logical_or_emitter::aux_vecs_count() const {
    return 2;
}
size_t jit_logical_or_emitter::aux_gprs_count() const {
    return 2;
}
void jit_logical_or_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}
template <cpu_isa_t isa>
void jit_logical_or_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                      const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == element::f32, "JIT Logical OR emitter supports only f32 precision");
    const VReg src0 = VReg(in_vec_idxs[0]);
    const VReg src1 = VReg(in_vec_idxs[1]);
    const VReg aux0 = VReg(aux_vec_idxs[0]);
    const VReg aux1 = VReg(aux_vec_idxs[1]);
    const VReg dst = VReg(out_vec_idxs[0]);
    auto one_reg = Reg(aux_gpr_idxs[0]);

    load_table_val("one", one_reg);

    h->vmv_v_x(aux0, x0);
    h->vmsne_vx(mask_vreg(), src0, x0);
    h->vmerge_vxm(aux0, aux0, one_reg);
    h->vmv_v_x(aux1, x0);
    h->vmsne_vx(mask_vreg(), src1, x0);
    h->vmerge_vxm(aux1, aux1, one_reg);
    h->vor_vv(dst, aux0, aux1);
}
std::set<std::vector<element::Type>> jit_logical_or_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}
void jit_logical_or_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

/// MAXIMUM ///
jit_maximum_emitter::jit_maximum_emitter(jit_generator_t* host, cpu_isa_t host_isa, const element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_maximum_emitter::jit_maximum_emitter(jit_generator_t* host,
                                         cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

size_t jit_maximum_emitter::get_inputs_num() const {
    return 2;
}

/// LESS EQUAL ///
jit_less_equal_emitter::jit_less_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                               ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                               ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_less_equal_emitter::jit_less_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                               ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                               const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

size_t jit_less_equal_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_less_equal_emitter::aux_fp_gprs_count() const {
    return 1;
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_less_equal_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                      const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);
    auto one = FReg(aux_fp_gpr_idxs[0]);
    load_table_val("one", one);

    h->vmfle_vv(mask_vreg(), src0, src1);    // compare "less than or equal", result in mask
    h->vmv_v_x(dst, zero);                   // set dst to 0
    h->vfadd_vf(dst, dst, one, VM::masked);  // set 1.0 where mask is true
}

void jit_less_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

std::set<std::vector<element::Type>> jit_less_equal_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_less_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

void jit_maximum_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for MAXIMUM");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_maximum_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    h->vfmax_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_maximum_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}
/// MINIMUM ///
jit_minimum_emitter::jit_minimum_emitter(jit_generator_t* host, cpu_isa_t host_isa, const element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_minimum_emitter::jit_minimum_emitter(jit_generator_t* host,
                                         cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

size_t jit_minimum_emitter::get_inputs_num() const {
    return 2;
}

void jit_minimum_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for MINIMUM");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_minimum_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    h->vfmin_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_minimum_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// LOGICAL AND ///
jit_logical_and_emitter::jit_logical_and_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_logical_and_emitter::jit_logical_and_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_and_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_logical_and_emitter::aux_fp_gprs_count() const {
    return 2;
}

size_t jit_logical_and_emitter::aux_vecs_count() const {
    return 2;
}

void jit_logical_and_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                        const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_logical_and_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);
    auto aux0 = VReg(aux_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[1]);

    auto fzero = FReg(aux_fp_gpr_idxs[0]);
    h->fmv_w_x(fzero, zero);

    auto fone = FReg(aux_fp_gpr_idxs[1]);
    load_table_val("one", fone);

    h->vmv_v_x(aux0, zero);
    h->vmfne_vf(mask_vreg(), src0, fzero);
    h->vfadd_vf(aux0, aux0, fone, VM::masked);

    h->vmv_v_x(aux1, zero);
    h->vmfne_vf(mask_vreg(), src1, fzero);
    h->vfadd_vf(aux1, aux1, fone, VM::masked);

    h->vand_vv(dst, aux0, aux1);
}

std::set<std::vector<element::Type>> jit_logical_and_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}
void jit_logical_and_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

/// LOGICAL NOT ///
jit_logical_not_emitter::jit_logical_not_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_logical_not_emitter::jit_logical_not_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_not_emitter::get_inputs_num() const {
    return 1;
}
size_t jit_logical_not_emitter::aux_fp_gprs_count() const {
    return 2;
}

void jit_logical_not_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                        const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_logical_not_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);
    auto fzero = FReg(aux_fp_gpr_idxs[0]);
    auto fone = FReg(aux_fp_gpr_idxs[1]);

    load_table_val("one", fone);
    h->fmv_w_x(fzero, zero);
    h->vmfne_vf(mask_vreg(), src, fzero);
    h->vfmv_v_f(dst, fone);
    h->vfsub_vf(dst, dst, fone, VM::masked);
}

std::set<std::vector<element::Type>> jit_logical_not_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_logical_not_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

/// LOGICAL XOR ///
jit_logical_xor_emitter::jit_logical_xor_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_logical_xor_emitter::jit_logical_xor_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                 ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                 ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_logical_xor_emitter::get_inputs_num() const {
    return 2;
}
size_t jit_logical_xor_emitter::aux_fp_gprs_count() const {
    return 2;
}
size_t jit_logical_xor_emitter::aux_vecs_count() const {
    return 2;
}

void jit_logical_xor_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                        const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_logical_xor_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                       const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);
    auto aux0 = VReg(aux_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[1]);

    auto fzero = FReg(aux_fp_gpr_idxs[0]);
    h->fmv_w_x(fzero, zero);

    auto fone = FReg(aux_fp_gpr_idxs[1]);
    load_table_val("one", fone);

    h->vmv_v_x(aux0, zero);
    h->vmfne_vf(mask_vreg(), src0, fzero);
    h->vfadd_vf(aux0, aux0, fone, VM::masked);

    h->vmv_v_x(aux1, zero);
    h->vmfne_vf(mask_vreg(), src1, fzero);
    h->vfadd_vf(aux1, aux1, fone, VM::masked);

    h->vxor_vv(dst, aux0, aux1);
}

std::set<std::vector<element::Type>> jit_logical_xor_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_logical_xor_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

/// MISH ///
jit_mish_emitter::jit_mish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    exp_emitter = std::make_unique<jit_exp_emitter>(host, host_isa, exec_prc);
    prepare_table();
}

jit_mish_emitter::jit_mish_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    exp_emitter = std::make_unique<jit_exp_emitter>(host, host_isa, exec_prc);
    prepare_table();
}

size_t jit_mish_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_mish_emitter::aux_gprs_count() const {
    return std::max<size_t>(exp_emitter->aux_gprs_count(), 1) + 1;
}

size_t jit_mish_emitter::aux_vecs_count() const {
    return std::max<size_t>(exp_emitter->aux_vecs_count() + 1, 2);
}

size_t jit_mish_emitter::aux_fp_gprs_count() const {
    return std::max<size_t>(exp_emitter->aux_fp_gprs_count(), 1);
}

void jit_mish_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_mish_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    // An equation other than mish(x) = x*tanh(srelu(x)) was used
    // to calculate mish, but it should be remembered that it is equivalent
    // equation, it uses the following rule:
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x),
    // hence the equation for mish can take the form:
    // mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1).
    // This option was chosen because computing tanh requires more registers
    // than exp, and also requires more constants to be stored in memory,
    // making the algorithm slower.

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    auto aux0 = VReg(aux_vec_idxs[0]);
    auto aux1 = VReg(aux_vec_idxs[1]);

    auto one = FReg(aux_fp_gpr_idxs[0]);
    auto tmpReg = Reg(aux_gpr_idxs[0]);

    load_table_val("fwd_mish_max_x_for_equation_f", aux1, tmpReg);
    h->vfmin_vv(aux1, src, aux1);

    auto exp_aux_vec_idxs = aux_vec_idxs;
    exp_aux_vec_idxs.erase(
        std::find(exp_aux_vec_idxs.begin(), exp_aux_vec_idxs.end(), static_cast<size_t>(aux1.getIdx())));
    exp_emitter->emit_code({static_cast<size_t>(aux1.getIdx())},
                           {static_cast<size_t>(aux1.getIdx())},
                           exp_aux_vec_idxs,
                           aux_gpr_idxs,
                           aux_fp_gpr_idxs);

    // save src as it may be the same as dst
    h->vmv_v_v(aux0, src);

    // (e^x+1)^2
    load_table_val("one", one);
    h->vfadd_vf(aux1, aux1, one);
    h->vfmul_vv(dst, aux1, aux1);

    h->vfsub_vf(aux1, dst, one);  // aux1 = (e^x+1)^2 - 1
    h->vfadd_vf(dst, dst, one);   // dst = (e^x+1)^2 + 1
    h->vfdiv_vv(dst, aux1, dst);
    h->vfmul_vv(dst, dst, aux0);
}

std::set<std::vector<element::Type>> jit_mish_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_mish_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
    push_arg_entry_of("fwd_mish_max_x_for_equation_f", 0x42317217);
}

void jit_mish_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_mul_add_emitter::jit_mul_add_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                         ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_mul_add_emitter::get_inputs_num() const {
    return 3;
}

void jit_mul_add_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto src2 = VReg(in_vec_idxs[2]);
    auto dst = VReg(out_vec_idxs[0]);

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

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

/// MUL ///
jit_multiply_emitter::jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_num() const {
    return 2;
}

void jit_multiply_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    h->vfmul_vv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// NEGATIVE ///
jit_negative_emitter::jit_negative_emitter(jit_generator_t* host, cpu_isa_t host_isa, const element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_negative_emitter::jit_negative_emitter(jit_generator_t* host,
                                           cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

size_t jit_negative_emitter::get_inputs_num() const {
    return 1;
}
void jit_negative_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel for NEGATIVE");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_negative_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    h->vfneg_vv(dst, src);
}

std::set<std::vector<element::Type>> jit_negative_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// NOT EQUAL ///
jit_not_equal_emitter::jit_not_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                             ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                             [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

jit_not_equal_emitter::jit_not_equal_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                             ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                             ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_not_equal_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_not_equal_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_not_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                      const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_not_equal_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    auto one = FReg(aux_fp_gpr_idxs[0]);
    load_table_val("one", one);

    h->vmfne_vv(mask_vreg(), src0, src1);
    h->vmv_v_x(dst, zero);
    h->vfadd_vf(dst, dst, one, VM::masked);
}

std::set<std::vector<element::Type>> jit_not_equal_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_not_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

/// PReLU ///
jit_prelu_emitter::jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

jit_prelu_emitter::jit_prelu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                     ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                     ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_prelu_emitter::get_inputs_num() const {
    return 2;
}

size_t jit_prelu_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_prelu_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);
    auto fzero = FReg(aux_fp_gpr_idxs[0]);

    if (src0.getIdx() != dst.getIdx()) {
        h->vmv_v_v(dst, src0);
    }

    h->fmv_w_x(fzero, zero);
    h->vmflt_vf(mask_vreg(), src0, fzero);

    h->vfmul_vv(dst, src0, src1, VM::masked);
}

std::set<std::vector<element::Type>> jit_prelu_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// ReLU ///
jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {
    if (const auto leaky_relu = ov::as_type_ptr<LeakyReluNode>(node)) {
        alpha = leaky_relu->get_slope();
    } else if (ov::is_type<ov::op::v0::Relu>(node)) {
        alpha = 0.F;
    } else {
        OV_CPU_JIT_EMITTER_THROW("Incompatible node!");
    }
    prepare_table();
}

jit_relu_emitter::jit_relu_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   float alpha,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      alpha(alpha) {
    prepare_table();
}

size_t jit_relu_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_relu_emitter::aux_fp_gprs_count() const {
    return 1;
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_relu_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);
    auto fzero = FReg(aux_fp_gpr_idxs[0]);
    h->fmv_w_x(fzero, zero);

    if (alpha == 0) {
        h->vfmax_vf(dst, src, fzero);
        return;
    }

    if (src.getIdx() != dst.getIdx()) {
        h->vmv_v_v(dst, src);
    }

    h->vmflt_vf(mask_vreg(), dst, fzero);

    FReg alpha_reg = fzero;
    load_table_val("alpha", alpha_reg);
    h->vfmul_vf(dst, dst, alpha_reg, VM::masked);
}

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_relu_emitter::register_table_entries() {
    if (alpha != 0) {
        push_arg_entry_of("alpha", dnnl::impl::float2int(alpha));
    }
}

/// Power Static ///
jit_power_static_emitter::jit_power_static_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                                   float power,
                                                   float scale,
                                                   float shift,
                                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      power(power),
      scale(scale),
      shift(shift) {
    prepare_table();
}

size_t jit_power_static_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_power_static_emitter::aux_gprs_count() const {
    if ((power == 0) || is_scale_shift() || (!is_sqrt() && !is_int_pow())) {
        return 2;
    }
    return 1;
}

bool jit_power_static_emitter::is_lmul_supported() const {
    return jit_emitter::is_lmul_supported() && (is_int_pow() || is_sqrt());
}

size_t jit_power_static_emitter::aux_vecs_count() const {
    if (is_scale_shift()) {
        return 2;
    }
    if (is_int_pow()) {
        return 1;
    }
    return 0;
}

size_t jit_power_static_emitter::aux_fp_gprs_count() const {
    return power < 0 ? 1 : 0;
}

void jit_power_static_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                         const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_power_static_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                        const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    if (power == 0) {
        auto tmp = Reg(aux_gpr_idxs[0]);
        load_table_val("one", dst, tmp);
        return;
    }

    if (is_scale_shift()) {
        auto aux0 = VReg(aux_vec_idxs[0]);
        auto aux1 = VReg(aux_vec_idxs[1]);
        auto tmp = Reg(aux_gpr_idxs[0]);
        load_table_val("shift", aux0, tmp);
        load_table_val("scale", aux1, tmp);
        h->vfmacc_vv(aux0, aux1, src);
        h->vmv_v_v(dst, aux0);
    } else {
        if (src.getIdx() != dst.getIdx()) {
            h->vmv_v_v(dst, src);
        }
    }

    // for power `-0.5f` there is `vfrsqrt7_v` instruction with worse accuracy
    if (is_sqrt()) {
        h->vfsqrt_v(dst, dst);

        if (power < 0) {
            auto one = FReg(aux_fp_gpr_idxs[0]);
            load_table_val("one", one);
            h->vfrdiv_vf(dst, dst, one);
        }
    } else if (is_int_pow()) {
        int64_t ipower = std::abs(static_cast<int64_t>(power)) - 1;

        auto aux0 = VReg(aux_vec_idxs[0]);
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
            auto one = FReg(aux_fp_gpr_idxs[0]);
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

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_power_static_emitter::register_table_entries() {
    if (scale != 1.F || shift != 0.F) {
        push_arg_entry_of("scale", dnnl::impl::float2int(scale));
        push_arg_entry_of("shift", dnnl::impl::float2int(shift));
    }
    if (power != 1.F) {
        push_arg_entry_of("power", dnnl::impl::float2int(power));
    }
    if (power < 0) {
        push_arg_entry_of("one", CONST_1_F);
    }
}

/// Sigmoid ///
jit_sigmoid_emitter::jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                         [[maybe_unused]] const std::shared_ptr<ov::Node>& node,
                                         ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      jit_exp_emitter_(std::make_unique<jit_exp_emitter>(host, host_isa, exec_prc)) {
    prepare_table();
}

jit_sigmoid_emitter::jit_sigmoid_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                         ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                         ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      jit_exp_emitter_(std::make_unique<jit_exp_emitter>(host, host_isa, exec_prc)) {
    prepare_table();
}

size_t jit_sigmoid_emitter::get_inputs_num() const {
    return 1;
}

size_t jit_sigmoid_emitter::aux_gprs_count() const {
    OV_CPU_JIT_EMITTER_ASSERT(jit_exp_emitter_, "JIT Exp emitter is missed!");
    return jit_exp_emitter_->aux_gprs_count() + 1;
}

size_t jit_sigmoid_emitter::aux_vecs_count() const {
    OV_CPU_JIT_EMITTER_ASSERT(jit_exp_emitter_, "JIT Exp emitter is missed!");
    return jit_exp_emitter_->aux_vecs_count() + 1;
}

size_t jit_sigmoid_emitter::aux_fp_gprs_count() const {
    OV_CPU_JIT_EMITTER_ASSERT(jit_exp_emitter_, "JIT Exp emitter is missed!");
    return std::max(jit_exp_emitter_->aux_fp_gprs_count(), 1LU);
}

void jit_sigmoid_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_sigmoid_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);
    auto sign_mask = VReg(aux_vec_idxs[aux_vecs_count() - 1]);
    auto aux = VReg(aux_vec_idxs[aux_vecs_count() - 2]);

    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.

    // we store the original sign and make x negative
    auto fzero = FReg(aux_fp_gpr_idxs[0]);
    h->vmfgt_vf(mask_vreg(), src, fzero);
    h->vfneg_vv(src, src, VM::masked);
    h->vmv1r_v(sign_mask, mask_vreg());  // save mask since exp uses mask too

    const auto exp_src_idxs = std::vector<size_t>{static_cast<size_t>(src.getIdx())};
    const auto exp_dst_idxs = std::vector<size_t>{static_cast<size_t>(dst.getIdx())};
    const auto exp_aux_vec_idxs =
        std::vector<size_t>{aux_vec_idxs.cbegin(), aux_vec_idxs.cbegin() + jit_exp_emitter_->aux_vecs_count()};
    jit_exp_emitter_->emit_code(exp_src_idxs, exp_dst_idxs, exp_aux_vec_idxs, aux_gpr_idxs, aux_fp_gpr_idxs);

    auto one = FReg(aux_fp_gpr_idxs[0]);
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
    h->vmv1r_v(mask_vreg(), sign_mask);  // pop mask
    h->vmerge_vvm(dst, dst, aux);
}

void jit_sigmoid_emitter::register_table_entries() {
    push_arg_entry_of("one", CONST_1_F);
}

void jit_sigmoid_emitter::emit_data() const {
    jit_emitter::emit_data();
    jit_exp_emitter_->emit_data();
}

std::set<std::vector<element::Type>> jit_sigmoid_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// SQRT ///
jit_sqrt_emitter::jit_sqrt_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_sqrt_emitter::jit_sqrt_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                   ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                   ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_sqrt_emitter::get_inputs_num() const {
    return 1;
}

void jit_sqrt_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                 const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_sqrt_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "Unsupported precision: ", exec_prc_);

    auto src = VReg(in_vec_idxs[0]);
    auto dst = VReg(out_vec_idxs[0]);

    h->vfsqrt_v(dst, src);
}

std::set<std::vector<element::Type>> jit_sqrt_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// SUB ///
jit_subtract_emitter::jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {}

jit_subtract_emitter::jit_subtract_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                           ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                           ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_subtract_emitter::get_inputs_num() const {
    return 2;
}

void jit_subtract_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                     const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == ov::intel_cpu::riscv64::cpu_isa_t::gv) {
        emit_isa<ov::intel_cpu::riscv64::cpu_isa_t::gv>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <ov::intel_cpu::riscv64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                    const std::vector<size_t>& out_vec_idxs) const {
    auto src0 = VReg(in_vec_idxs[0]);
    auto src1 = VReg(in_vec_idxs[1]);
    auto dst = VReg(out_vec_idxs[0]);

    switch (exec_prc_) {
    case ov::element::f32:
        h->vfsub_vv(dst, src0, src1);
        break;
    case ov::element::i32:
        h->vsub_vv(dst, src0, src1);
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Unsupported precision: ", exec_prc_);
    }
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}, {element::i32, element::i32}};
}

#undef CONST_1_F

}  // namespace ov::intel_cpu::riscv64
