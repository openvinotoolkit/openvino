// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"

#include <memory>
#include "common/utils.hpp"
#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

namespace {
ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(
            input.get_source_output().get_element_type());
    }

    assert(std::all_of(
        input_precisions.begin(),
        input_precisions.end(),
        [&input_precisions](const ov::element::Type& precision) {return precision == input_precisions[0]; }));

    return input_precisions[0];
}
} // namespace

/// ABS ///
jit_abs_emitter::jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_abs_emitter::jit_abs_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_abs_emitter::get_inputs_count() const { return 1; }

void jit_abs_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_abs_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fabs(dst.s, src.s);
}

std::set<std::vector<element::Type>> jit_abs_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// ADD ///
jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
                                 : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_add_emitter::jit_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_add_emitter::get_inputs_count() const { return 2; }

void jit_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fadd(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// CLAMP ///
jit_clamp_emitter::jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
                                     : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    const auto clamp = std::dynamic_pointer_cast<ov::op::v0::Clamp>(node);
    if (clamp == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to ov::op::v0::Clamp");
    }
    min = static_cast<float>(clamp->get_min());
    max = static_cast<float>(clamp->get_max());

    prepare_table();
}

jit_clamp_emitter::jit_clamp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float min,
                                     const float max,
                                     const ov::element::Type exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc),
                                       min(min),
                                       max(max) {
    prepare_table();
}

size_t jit_clamp_emitter::get_inputs_count() const { return 1; }

size_t jit_clamp_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_clamp_emitter::get_aux_gprs_count() const { return 1; }

void jit_clamp_emitter::register_table_entries() {
    push_arg_entry_of("min", dnnl::impl::float2int(min), true);
    push_arg_entry_of("max", dnnl::impl::float2int(max), true);
}

void jit_clamp_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_clamp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg aux = TReg(aux_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->ld1r(aux.s, table_val2("min"));
    h->fmax(dst.s, src.s, aux.s);
    h->ld1r(aux.s, table_val2("max"));
    h->fmin(dst.s, dst.s, aux.s);
}

std::set<std::vector<element::Type>> jit_clamp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// DIVIDE ///
jit_divide_emitter::jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_divide_emitter::jit_divide_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_divide_emitter::get_inputs_count() const { return 2; }

void jit_divide_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_divide_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fdiv(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// EQUAL ///
jit_equal_emitter::jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
                                     : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}
jit_equal_emitter::jit_equal_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const ov::element::Type exec_prc)
                                     : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_equal_emitter::get_inputs_count() const { return 2; }

size_t jit_equal_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_equal_emitter::get_aux_gprs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_equal_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

void jit_equal_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg src1 = TReg(in_vec_idxs[0]);
    const TReg src2 = TReg(in_vec_idxs[1]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    h->fcmeq(dst.s, src1.s, src2.s);

    h->ld1r(aux.s, table_val2("one"));
    h->and_(dst.b16, dst.b16, aux.b16);
}

void jit_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", 0x3f800000, true);
}

/// EXPONENT ///
jit_exp_emitter::jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_exp_emitter::jit_exp_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_exp_emitter::get_inputs_count() const { return 1; }

size_t jit_exp_emitter::get_aux_vecs_count() const { return 4; }

size_t jit_exp_emitter::get_aux_gprs_count() const { return 1; }

void jit_exp_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_exp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != ov::element::f32) {
        OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(aux_vec_idxs[2]);

    const TReg vmm_mask(aux_vec_idxs[3]);

    h->ld1r(vmm_aux0.s, table_val2("exp_ln_flt_max_f"));
    h->fmin(vmm_dst.s, vmm_src.s, vmm_aux0.s);
    h->ld1r(vmm_aux0.s, table_val2("exp_ln_flt_min_f"));
    h->fmax(vmm_dst.s, vmm_dst.s, vmm_aux0.s);

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->fcmgt(vmm_mask.s, vmm_src.s, vmm_aux0.s);
    h->mov(vmm_aux1.b16, vmm_dst.b16);

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->ld1r(vmm_aux0.s, table_val2("exp_log2ef"));
    h->ld1r(vmm_aux2.s, table_val2("half"));
    h->fmla(vmm_aux2.s, vmm_dst.s, vmm_aux0.s);

    // tmp = floorf(fx)
    h->frintm(vmm_aux2.s, vmm_aux2.s);

    // keep vmm_src = fx for further computations
    h->mov(vmm_dst.b16, vmm_aux2.b16);

    // x = x - fx * ln2
    h->ld1r(vmm_aux0.s, table_val2("ln2f"));
    h->fmls(vmm_aux1.s, vmm_aux2.s, vmm_aux0.s);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->ld1r(vmm_aux0.s, table_val2("one"));
    h->fsub(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
    h->fcvtzs(vmm_aux2.s, vmm_dst.s);

    h->ld1r(vmm_aux0.s, table_val2("exponent_bias"));
    h->add(vmm_aux2.s, vmm_aux2.s, vmm_aux0.s);

    const int n_mantissa_bits = 23;
    h->sqshl(vmm_aux2.s, vmm_aux2.s, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    h->and_(vmm_aux2.b16, vmm_mask.b16, vmm_aux2.b16);

    // compute polynomial
    h->ld1r(vmm_aux0.s, table_val2("exp_pol5"));
    h->ld1r(vmm_dst.s, table_val2("exp_pol4"));
    h->fmla(vmm_dst.s, vmm_aux1.s, vmm_aux0.s);

    h->ld1r(vmm_aux0.s, table_val2("exp_pol3"));
    h->fmla(vmm_aux0.s, vmm_dst.s, vmm_aux1.s);

    h->ld1r(vmm_dst.s, table_val2("exp_pol2"));
    h->fmla(vmm_dst.s, vmm_aux0.s, vmm_aux1.s);

    h->ld1r(vmm_aux0.s, table_val2("exp_pol1"));
    h->fmla(vmm_aux0.s, vmm_dst.s, vmm_aux1.s);

    h->ld1r(vmm_dst.s, table_val2("one"));
    h->fmla(vmm_dst.s, vmm_aux0.s, vmm_aux1.s);

    // y = y * 2^n
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_aux2.s);
    h->ld1r(vmm_aux0.s, table_val2("two"));
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_aux0.s);
}

void jit_exp_emitter::register_table_entries() {
    push_arg_entry_of("exp_ln_flt_max_f", 0x42b17218, true);
    push_arg_entry_of("exp_ln_flt_min_f", 0xc2aeac50, true);
    push_arg_entry_of("exp_log2ef", 0x3fb8aa3b, true);
    push_arg_entry_of("one", 0x3f800000, true);
    push_arg_entry_of("two", 0x40000000, true);
    push_arg_entry_of("half", 0x3f000000, true);
    push_arg_entry_of("ln2f", 0x3f317218, true);
    push_arg_entry_of("exponent_bias", 0x0000007f, true);
    push_arg_entry_of("exp_pol1", 0x3f7ffffb, true);
    push_arg_entry_of("exp_pol2", 0x3efffee3, true);
    push_arg_entry_of("exp_pol3", 0x3e2aad40, true);
    push_arg_entry_of("exp_pol4", 0x3d2b9d0d, true);
    push_arg_entry_of("exp_pol5", 0x3c07cfce, true);
}

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// MUL_ADD ///
jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
                                         : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_mul_add_emitter::jit_mul_add_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc)
                                         : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_mul_add_emitter::get_inputs_count() const { return 3; }

size_t jit_mul_add_emitter::get_aux_vecs_count() const { return 1; }

void jit_mul_add_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg dst = TReg(out_vec_idxs[0]);

    TReg mul0(in_vec_idxs[0]);
    if (dst.getIdx() == in_vec_idxs[0]) {
        TReg aux(aux_vec_idxs[0]);
        TReg src0(in_vec_idxs[0]);
        h->mov(aux.b16, src0.b16);
        mul0 = aux;
    }

    TReg mul1(in_vec_idxs[1]);
    if (dst.getIdx() == in_vec_idxs[1]) {
        TReg aux(aux_vec_idxs[0]);
        TReg src1(in_vec_idxs[1]);
        h->mov(aux.b16, src1.b16);
        mul1 = aux;
    }

    if (dst.getIdx() != in_vec_idxs[2]) {
        TReg src2(in_vec_idxs[2]);
        h->mov(dst.b16, src2.b16);
    }

    h->fmla(dst.s, mul0.s, mul1.s);
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

/// MULTIPLY ///
jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {}

jit_multiply_emitter::jit_multiply_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc)
                                           : jit_emitter(host, host_isa, exec_prc) {}

size_t jit_multiply_emitter::get_inputs_count() const { return 2; }

void jit_multiply_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fmul(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// POWER ///
jit_power_static_emitter::jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                   const std::shared_ptr<ov::Node>& node,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, host_isa, node, exec_prc) {
    auto powerStaticNode = ov::as_type_ptr<ov::snippets::op::PowerStatic>(node);
    if (powerStaticNode == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to snippets::op::PowerStatic");
    }

    power = powerStaticNode->get_power();
    scale = 1.f;
    shift = 0.f;

    prepare_table();
}

jit_power_static_emitter::jit_power_static_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                   const float power,
                                                   const float scale,
                                                   const float shift,
                                                   const ov::element::Type exec_prc)
                                                   : jit_emitter(host, host_isa, exec_prc),
                                                     power(power),
                                                     scale(scale),
                                                     shift(shift) {
    prepare_table();
}

size_t jit_power_static_emitter::get_inputs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_vecs_count() const { return 1; }

size_t jit_power_static_emitter::get_aux_gprs_count() const { return 2; }

void jit_power_static_emitter::register_table_entries() {
    push_arg_entry_of("power", dnnl::impl::float2int(power), true);
    push_arg_entry_of("scale", dnnl::impl::float2int(scale), true);
    push_arg_entry_of("shift", dnnl::impl::float2int(shift), true);
}

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_power_static_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_power_static_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg dst = TReg(out_vec_idxs[0]);

    if (power == 0.f) {
        h->fmov(dst.s, 1.);
        return;
    }

    bool get_from_dst = false;
    const auto src = [&in_vec_idxs, &out_vec_idxs, &get_from_dst]() -> TReg {
        return get_from_dst ? TReg(out_vec_idxs[0]) : TReg(in_vec_idxs[0]);
    };

    TReg aux = TReg(aux_vec_idxs[0]);
    if (scale != 1.f) {
        auto adr = table_val2("scale");
        h->ld1r(aux.s, adr);
        h->fmul(dst.s, src().s, aux.s);
        get_from_dst = true;
    }

    if (shift != 0.f) {
        auto adr = table_val2("shift");
        h->ld1r(aux.s, adr);
        h->fadd(dst.s, src().s, aux.s);
        get_from_dst = true;
    }

    if (power == 1.f) {
        if (!get_from_dst && (in_vec_idxs[0] != dst.getIdx())) {
            h->mov(dst.b16, src().b16);
        }
        return;
    }

    if (std::floor(power) == power && power > 0) {
        h->mov(aux.b16, src().b16);
        h->fmov(dst.s, 1.);

        auto current_power = static_cast<size_t>(power);
        while (current_power > 0) {
            if (current_power & 1) {
                h->fmul(dst.s, dst.s, aux.s);
            }
            if (current_power > 1) {
                h->fmul(aux.s, aux.s, aux.s);
            }
            current_power = current_power >> 1;
        }
    } else {
        auto pow_f32_addr = reinterpret_cast<uintptr_t>(::powf);

        Xbyak_aarch64::XReg func_reg(aux_gpr_idxs[0]);
        h->mov(func_reg, pow_f32_addr);

        Xbyak_aarch64::SReg s0(0);
        Xbyak_aarch64::SReg s1(1);

        const std::unordered_set<size_t> exclude = {src().getIdx(), dst.getIdx()};
        store_context(exclude);
        for (auto i = 0; i < 4; i++) {
            h->mov(s0, src().s[i]);
            h->ldr(s1, table_val("power"));

            h->str(Xbyak_aarch64::QReg(dst.getIdx()), pre_ptr(h->sp, -16));
            h->str(Xbyak_aarch64::QReg(src().getIdx()), pre_ptr(h->sp, -16));
            h->blr(func_reg);
            h->ldr(Xbyak_aarch64::QReg(src().getIdx()), post_ptr(h->sp, 16));
            h->ldr(Xbyak_aarch64::QReg(dst.getIdx()), post_ptr(h->sp, 16));

            Xbyak_aarch64::WReg w0(0);
            h->fmov(w0, s0);
            h->mov(dst.s[i], w0);
        }
        restore_context(exclude);
    }
}

/// PRELU ///
jit_prelu_emitter::jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_prelu_emitter::jit_prelu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_prelu_emitter::get_inputs_count() const { return 2; }

size_t jit_prelu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_prelu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_prelu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_prelu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[0]);
    TReg src2 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fcmge(dst.s, src1.s, 0.0);
    h->fmul(tmp.s, src1.s, src2.s);
    h->bsl(dst.b16, src1.b16, tmp.b16);
}

/// RELU ///
jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_relu_emitter::jit_relu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_relu_emitter::get_inputs_count() const { return 1; }

size_t jit_relu_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_relu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_relu_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_relu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->movi(tmp.s, 0);
    h->fmaxnm(dst.s, src.s, tmp.s);
}

/// SELECT ///
jit_select_emitter::jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node)
                                       : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
}
jit_select_emitter::jit_select_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const ov::element::Type exec_prc)
                                       : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_select_emitter::get_inputs_count() const { return 3; }

size_t jit_select_emitter::get_aux_vecs_count() const { return 1; }

std::set<std::vector<element::Type>> jit_select_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32, element::f32}};
}

void jit_select_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_select_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg src1 = TReg(in_vec_idxs[0]);
    const TReg src2 = TReg(in_vec_idxs[1]);
    const TReg src3 = TReg(in_vec_idxs[2]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    h->eor(aux.b16, aux.b16, aux.b16);
    h->fcmgt(aux.s, src1.s, aux.s);

    h->bsl(aux.b16, src2.b16, src3.b16);
    h->mov(dst.b16, aux.b16);
}

/// SIGMOID ///
jit_sigmoid_emitter::jit_sigmoid_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, node);
}

jit_sigmoid_emitter::jit_sigmoid_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_sigmoid_emitter::get_inputs_count() const { return 1; }

size_t jit_sigmoid_emitter::get_aux_vecs_count() const {
    return exp_emitter->get_aux_vecs_count() + 2;
}

size_t jit_sigmoid_emitter::get_aux_gprs_count() const {
    return exp_emitter->get_aux_gprs_count() + 1;
}

void jit_sigmoid_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_sigmoid_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (exec_prc_ != ov::element::f32) {
        OPENVINO_THROW("unsupported precision: " + exec_prc_.to_string());
    }

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);

    const TReg vmm_aux0(aux_vec_idxs[exp_emitter->get_aux_vecs_count() + 1]);
    const TReg vmm_mask(aux_vec_idxs[exp_emitter->get_aux_vecs_count()]);

    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_mask for the mask as exp_compute does not use it.
    // we store the original sign and make x negative
    h->eor(vmm_aux0.b16, vmm_aux0.b16, vmm_aux0.b16);
    h->fcmgt(vmm_mask.s, vmm_src.s, vmm_aux0.s);

    h->ld1r(vmm_aux0.s, table_val2("sign_mask"));
    h->orr(vmm_aux0.b16, vmm_src.b16, vmm_aux0.b16);

    exp_emitter->emit_code(
            { vmm_aux0.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    // dup exp(x)
    h->mov(vmm_aux1.b16, vmm_dst.b16);
    // (exp(x) + 1)
    h->ld1r(vmm_aux0.s, table_val2("one"));
    h->fadd(vmm_aux1.s, vmm_aux1.s, vmm_aux0.s);
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_dst.s, vmm_dst.s, vmm_aux1.s);

    // Now we have to apply the "symmetry" based on original sign
    h->ld1r(vmm_aux2.s, table_val2("one"));
    h->fsub(vmm_aux2.s, vmm_aux2.s, vmm_dst.s);

    h->bsl(vmm_mask.b16, vmm_aux2.b16, vmm_dst.b16);
    h->mov(vmm_dst.b16, vmm_mask.b16);
}

void jit_sigmoid_emitter::register_table_entries() {
    push_arg_entry_of("one", 0x3f800000, true);
    push_arg_entry_of("sign_mask", 0x80000000, true);
}

void jit_sigmoid_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_sigmoid_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// SUBTRACT ///
jit_subtract_emitter::jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
                                           : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_subtract_emitter::jit_subtract_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_subtract_emitter::get_inputs_count() const { return 2; }

void jit_subtract_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_subtract_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fsub(dst.s, src0.s, src1.s);
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32, element::f32}};
}

/// SWISH ///
jit_swish_emitter::jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, node);
}

jit_swish_emitter::jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const float beta,
                                     const ov::element::Type exec_prc)
        : jit_emitter(host, host_isa, exec_prc), beta(beta) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, exec_prc);
}

size_t jit_swish_emitter::get_inputs_count() const {return 1; }

size_t jit_swish_emitter::get_aux_vecs_count() const {
    return sigmoid_emitter->get_aux_vecs_count() + 2;
}

size_t jit_swish_emitter::get_aux_gprs_count() const {
    return sigmoid_emitter->get_aux_gprs_count() + 1;
}

void jit_swish_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_swish_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_orig_src(aux_vec_idxs[sigmoid_emitter->get_aux_vecs_count()]);
    const TReg vmm_aux(aux_vec_idxs[sigmoid_emitter->get_aux_vecs_count() + 1]);

    h->mov(vmm_orig_src.b16, vmm_src.b16);

    // x*beta
    h->ld1r(vmm_aux.s, table_val2("beta"));
    h->fmul(vmm_aux.s, vmm_aux.s, vmm_src.s);

    // sigmoid(x*beta)
    sigmoid_emitter->emit_code(
            { vmm_aux.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    // x*sigmoid(x*beta)
    h->fmul(vmm_dst.s, vmm_dst.s, vmm_orig_src.s);
}

void jit_swish_emitter::register_table_entries() {
    push_arg_entry_of("beta", dnnl::impl::float2int(beta), true);
}

void jit_swish_emitter::emit_data() const {
    jit_emitter::emit_data();
    sigmoid_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_swish_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// TANH ///
jit_tanh_emitter::jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
                                   : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, node);
}

jit_tanh_emitter::jit_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc)
                                   : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    sigmoid_emitter = std::make_unique<jit_sigmoid_emitter>(h, host_isa, exec_prc);
}

size_t jit_tanh_emitter::get_inputs_count() const { return 1; }

size_t jit_tanh_emitter::get_aux_vecs_count() const {
    return sigmoid_emitter->get_aux_vecs_count() + 1;
}

size_t jit_tanh_emitter::get_aux_gprs_count() const {
    return sigmoid_emitter->get_aux_gprs_count() + 1;
}

void jit_tanh_emitter::emit_impl(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_tanh_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    TReg aux = TReg(aux_vec_idxs.back());

    h->ld1r(aux.s, table_val2("two"));
    h->uni_fmul(aux.s, src.s, aux.s);

    sigmoid_emitter->emit_code(
            { aux.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    h->ld1r(aux.s, table_val2("two"));
    h->uni_fmul(dst.s, aux.s, dst.s);
    h->ld1r(aux.s, table_val2("one"));
    h->uni_fsub(dst.s, dst.s, aux.s);
}

void jit_tanh_emitter::register_table_entries() {
    push_arg_entry_of("one", 0x3f800000, true);
    push_arg_entry_of("two", 0x40000000, true);
}

void jit_tanh_emitter::emit_data() const {
    jit_emitter::emit_data();
    sigmoid_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_tanh_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
