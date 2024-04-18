// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_eltwise_emitters.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"

#include <memory>
#include <cmath>
#include "common/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

#define OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)                        \
    OV_CPU_JIT_EMITTER_ASSERT(                                                \
        ((exec_prc_ == ov::element::f16) || (exec_prc_ == ov::element::f32)), \
        "unsupported precision: " + exec_prc_.to_string());                   \

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

uint32_t float2int(const float value, const ov::element::Type& type = ov::element::f32) {
    if (type == ov::element::f16) {
        return dnnl::impl::utils::bit_cast<int16_t, float16>(value);
    } else if (type == ov::element::f32) {
        return dnnl::impl::utils::bit_cast<uint32_t, float>(value);
    } else {
        OV_CPU_JIT_EMITTER_ASSERT(false, "unsupported precision: " + type.to_string());
    }
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_abs_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fabs(dst, src);
}

std::set<std::vector<element::Type>> jit_abs_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fadd(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
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
    push_arg_entry_of("min", float2int(min, exec_prc_), true, exec_prc_);
    push_arg_entry_of("max", float2int(max, exec_prc_), true, exec_prc_);
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_clamp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg aux = TReg(aux_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->ld1r(aux, table_val2("min"));
    h->fmax(dst, src, aux);
    h->ld1r(aux, table_val2("max"));
    h->fmin(dst, dst, aux);
}

std::set<std::vector<element::Type>> jit_clamp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_divide_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fdiv(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_divide_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
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
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_equal_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg src1 = TReg(in_vec_idxs[0]);
    const TReg src2 = TReg(in_vec_idxs[1]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    h->fcmeq(dst, src1, src2);

    h->ld1r(aux, table_val2("one"));
    h->and_(BReg(dst.getIdx()), BReg(dst.getIdx()), BReg(aux.getIdx()));
}

void jit_equal_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true, exec_prc_);
}

/// ELU ///
jit_elu_emitter::jit_elu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, get_arithmetic_binary_exec_precision(node)) {
    const auto elu = std::dynamic_pointer_cast<ov::op::v0::Elu>(node);
    if (elu == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to ov::op::v0::Clamp");
    }
    alpha = static_cast<float>(elu->get_alpha());

    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, node);
}

jit_elu_emitter::jit_elu_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const float alpha,
                                 const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc), alpha(alpha) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_elu_emitter::get_inputs_count() const { return 1; }

size_t jit_elu_emitter::get_aux_vecs_count() const {
    return std::max(exp_emitter->get_aux_vecs_count() + 1ull, 2ull);
}

size_t jit_elu_emitter::get_aux_gprs_count() const {
    return exp_emitter->get_aux_gprs_count() + 1;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_elu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;

    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[std::max<size_t>(exp_emitter->get_aux_vecs_count(), 1)]);

    h->mov(BReg(vmm_aux1.getIdx()), BReg(vmm_src.getIdx()));

    // compute exponent
    exp_emitter->emit_code(
            { vmm_src.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    // alpha * (exp(x) - 1)
    const TReg vmm_aux0(aux_vec_idxs[0]);
    h->ld1r(vmm_aux0, table_val2("one"));
    h->fsub(vmm_dst, vmm_dst, vmm_aux0);
    h->ld1r(vmm_aux0, table_val2("alpha"));
    h->fmul(vmm_aux0, vmm_dst, vmm_aux0);

    // combine with mask
    h->fcmgt(vmm_dst, vmm_aux1, 0.f);
    h->bsl(BReg(vmm_dst.getIdx()), BReg(vmm_aux1.getIdx()), BReg(vmm_aux0.getIdx()));
}

void jit_elu_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("alpha", float2int(alpha, exec_prc_), true, exec_prc_);
}

void jit_elu_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_elu_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
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

size_t jit_exp_emitter::get_aux_vecs_count() const { return 5; }

size_t jit_exp_emitter::get_aux_gprs_count() const { return 1; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_exp_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    const auto min = std::numeric_limits<float16>::min();
    const auto max = std::numeric_limits<float16>::max();

    { // NOLINT
    auto exec_prc = sizeof(type) == 4 ? ov::element::f32 : ov::element::f16;
    //convert(ov::element::f32, ov::element::f16, in_vec_idxs, out_vec_idxs, aux_vec_idxs);

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(aux_vec_idxs[2]);
    const TReg vmm_mask(aux_vec_idxs[3]);

    // source and destination registers can be the same:
    // use vmm_aux2 to store destination before get mask
    h->ld1r(vmm_aux0, table_val2("exp_ln_flt_max_f", exec_prc));
    h->fmin(vmm_aux2, vmm_src, vmm_aux0);

    h->ld1r(vmm_aux0, table_val2("exp_ln_flt_min_f", exec_prc));
    // get mask of values lower than log(FLT_MIN) to zero them in the output
    h->fcmgt(vmm_mask, vmm_src, vmm_aux0);
    h->fmax(vmm_dst, vmm_aux2, vmm_aux0);

    h->mov(BReg(vmm_aux1.getIdx()), BReg(vmm_dst.getIdx()));

    // calculate exp(x)
    // fx = x * log2ef + 0.5
    h->ld1r(vmm_aux0, table_val2("exp_log2ef", exec_prc));
    h->ld1r(vmm_aux2, table_val2("half", exec_prc));
    h->fmla(vmm_aux2, vmm_dst, vmm_aux0);

    // tmp = floorf(fx)
    h->frintm(vmm_aux2, vmm_aux2);

    // keep vmm_src = fx for further computations
    h->mov(BReg(vmm_dst.getIdx()), BReg(vmm_aux2.getIdx()));

    // x = x - fx * ln2
    h->ld1r(vmm_aux0, table_val2("ln2f", exec_prc));
    h->fmls(vmm_aux1, vmm_aux2, vmm_aux0);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->ld1r(vmm_aux0, table_val2("one", exec_prc));
    h->fsub(vmm_dst, vmm_dst, vmm_aux0);

    //convert(ov::element::f16, ov::element::f32, in_vec_idxs, out_vec_idxs, aux_vec_idxs);
    }

    if (exec_prc_ == ov::element::f16) {
        auto exec_prc = ov::element::f32;

        //convert(ov::element::f32, ov::element::f16, in_vec_idxs, out_vec_idxs, aux_vec_idxs);

        using TReg = typename cpu_isa_vector_traits<isa, float>::TReg;
        using BReg = typename cpu_isa_vector_traits<isa, float>::BReg;
        const TReg vmm_dst(out_vec_idxs[0]);
        const TReg vmm_aux2(aux_vec_idxs[1]);
        const TReg vmm_aux0(aux_vec_idxs[2]);
        const TReg vmm_aux3(aux_vec_idxs[4]);

        typedef Xbyak_aarch64::VReg VReg;
        h->fcvtl(VReg(vmm_aux0.getIdx()).s4, VReg(vmm_dst.getIdx()).h4);
        h->fcvtzs(vmm_aux2, vmm_aux0); // <= this should be in fp32

        h->ld1r(vmm_aux0, table_val2("exponent_bias", exec_prc));
        h->add(vmm_aux2, vmm_aux2, vmm_aux0);

        const int n_mantissa_bits = exec_prc == ov::element::f16 ? 10 : 23;
        h->sqshl(vmm_aux2, vmm_aux2, n_mantissa_bits); // <= this should be in fp32
        h->fcvtn(VReg(vmm_aux2.getIdx()).h4, VReg(vmm_aux2.getIdx()).s4);



        h->fcvtl2(VReg(vmm_aux0.getIdx()).s4, VReg(vmm_dst.getIdx()).h8);
        h->fcvtzs(vmm_aux3, vmm_aux0); // <= this should be in fp32

        h->ld1r(vmm_aux0, table_val2("exponent_bias", exec_prc));
        h->add(vmm_aux3, vmm_aux3, vmm_aux0);

        h->sqshl(vmm_aux3, vmm_aux3, n_mantissa_bits); // <= this should be in fp32
        h->fcvtn2(VReg(vmm_aux2.getIdx()).h8, VReg(vmm_aux3.getIdx()).s4);
    } else {
        auto exec_prc = ov::element::f32;

        using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
        using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
        const TReg vmm_src(in_vec_idxs[0]);
        const TReg vmm_dst(out_vec_idxs[0]);
        const TReg vmm_aux1(aux_vec_idxs[0]);
        const TReg vmm_aux2(aux_vec_idxs[1]);
        const TReg vmm_aux0(aux_vec_idxs[2]);
        const TReg vmm_mask(aux_vec_idxs[3]);

        h->fcvtzs(vmm_aux2, vmm_dst); // <= this should be in fp32

        h->ld1r(vmm_aux0, table_val2("exponent_bias", exec_prc));
        h->add(vmm_aux2, vmm_aux2, vmm_aux0);

        const int n_mantissa_bits = exec_prc == ov::element::f16 ? 10 : 23;
        h->sqshl(vmm_aux2, vmm_aux2, n_mantissa_bits); // <= this should be in fp32
    }

    {
    auto exec_prc = sizeof(type) == 4 ? ov::element::f32 : ov::element::f16;
    //convert(ov::element::f32, ov::element::f16, in_vec_idxs, out_vec_idxs, aux_vec_idxs);

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    const TReg vmm_aux0(aux_vec_idxs[2]);
    const TReg vmm_mask(aux_vec_idxs[3]);

    // set zeroes at those points which were < log(FLT_MIN)
    h->and_(BReg(vmm_aux2.getIdx()), BReg(vmm_mask.getIdx()), BReg(vmm_aux2.getIdx()));

    // compute polynomial
    h->ld1r(vmm_aux0, table_val2("exp_pol5", exec_prc));
    h->ld1r(vmm_dst, table_val2("exp_pol4", exec_prc));
    h->fmla(vmm_dst, vmm_aux1, vmm_aux0);

    h->ld1r(vmm_aux0, table_val2("exp_pol3", exec_prc));
    h->fmla(vmm_aux0, vmm_dst, vmm_aux1);

    h->ld1r(vmm_dst, table_val2("exp_pol2", exec_prc));
    h->fmla(vmm_dst, vmm_aux0, vmm_aux1);

    h->ld1r(vmm_aux0, table_val2("exp_pol1", exec_prc));
    h->fmla(vmm_aux0, vmm_dst, vmm_aux1);

    h->ld1r(vmm_dst, table_val2("one", exec_prc));
    h->fmla(vmm_dst, vmm_aux0, vmm_aux1);

    // y = y * 2^n
    h->fmul(vmm_dst, vmm_dst, vmm_aux2);
    h->ld1r(vmm_aux0, table_val2("two", exec_prc));
    h->fmul(vmm_dst, vmm_dst, vmm_aux0);

    //convert(ov::element::f16, ov::element::f32, in_vec_idxs, out_vec_idxs, aux_vec_idxs);
    }
}

void jit_exp_emitter::register_table_entries() {
    const auto exec_prc = ov::element::f16;

    const auto push_args_entry_of = [&](const ov::element::Type& exec_prc) {
        const auto exec_prc_str = "_" + exec_prc.to_string();
        push_arg_entry_of("exp_ln_flt_max_f" + exec_prc_str, float2int(std::log(FLT_MAX), exec_prc), true, exec_prc);
        push_arg_entry_of("exp_ln_flt_min_f" + exec_prc_str, float2int(std::log(FLT_MIN), exec_prc), true, exec_prc);
        push_arg_entry_of("exp_log2ef" + exec_prc_str, float2int(1.44269502, exec_prc), true, exec_prc);
        push_arg_entry_of("one" + exec_prc_str, float2int(1.f, exec_prc), true, exec_prc);
        push_arg_entry_of("two" + exec_prc_str, float2int(2.f, exec_prc), true, exec_prc);
        push_arg_entry_of("half" + exec_prc_str, float2int(0.5f, exec_prc), true, exec_prc);
        push_arg_entry_of("ln2f" + exec_prc_str, float2int(std::log(2.f), exec_prc), true, exec_prc);
        push_arg_entry_of("exponent_bias" + exec_prc_str, exec_prc == ov::element::f32 ? 0x0000007f : 0x000F, true, exec_prc);
        push_arg_entry_of("exp_pol1" + exec_prc_str, float2int(0.999999701f, exec_prc), true, exec_prc);
        push_arg_entry_of("exp_pol2" + exec_prc_str, float2int(0.499991506f, exec_prc), true, exec_prc);
        push_arg_entry_of("exp_pol3" + exec_prc_str, float2int(0.166676521f, exec_prc), true, exec_prc);
        push_arg_entry_of("exp_pol4" + exec_prc_str, float2int(0.0418978221f, exec_prc), true, exec_prc);
        push_arg_entry_of("exp_pol5" + exec_prc_str, float2int(0.00828929059f, exec_prc), true, exec_prc);
    };

    push_args_entry_of(ov::element::f16);
    push_args_entry_of(ov::element::f32);
}

std::set<std::vector<element::Type>> jit_exp_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// Floor ///
jit_floor_emitter::jit_floor_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_floor_emitter::jit_floor_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_floor_emitter::get_inputs_count() const { return 1; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_floor_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_);

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);
    h->frintm(dst, src);
}

std::set<std::vector<element::Type>> jit_floor_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// GELU_ERF ///
jit_gelu_erf_emitter::jit_gelu_erf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, node);
}

jit_gelu_erf_emitter::jit_gelu_erf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                           dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                           const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_gelu_erf_emitter::get_inputs_count() const { return 1; }

size_t jit_gelu_erf_emitter::get_aux_vecs_count() const {
    return std::max<size_t>(exp_emitter->get_aux_vecs_count() + 3, 7);
}

size_t jit_gelu_erf_emitter::get_aux_gprs_count() const {
    return exp_emitter->get_aux_gprs_count() + 1;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_gelu_erf_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;

    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);

    const TReg vmm_aux0(aux_vec_idxs[0]);
    const TReg vmm_aux1(aux_vec_idxs[1]);
    const TReg vmm_aux2(aux_vec_idxs[2]);
    const TReg vmm_aux3(aux_vec_idxs[3]);
    const TReg vmm_aux(aux_vec_idxs[std::max<size_t>(exp_emitter->get_aux_vecs_count(), 4)]);
    const TReg vmm_aux_t(aux_vec_idxs[std::max<size_t>(exp_emitter->get_aux_vecs_count() + 1, 5)]);
    const TReg vmm_aux_dst(aux_vec_idxs[std::max<size_t>(exp_emitter->get_aux_vecs_count() + 2, 6)]);

    // x = s / sqrt(2)
    h->ld1r(vmm_aux0, table_val2("gelu_erf_one_over_sqrt_two"));
    h->fmul(vmm_aux0, vmm_aux0, vmm_src);

    // abs(x)
    h->fabs(vmm_aux0, vmm_aux0);

    // t = 1 / (p*x + 1)
    h->ld1r(vmm_aux1, table_val2("gelu_erf_approx_const"));
    h->ld1r(vmm_aux2, table_val2("one"));
    h->mov(BReg(vmm_aux3.getIdx()), BReg(vmm_aux2.getIdx()));
    h->fmla(vmm_aux2, vmm_aux1, vmm_aux0);
    h->fdiv(vmm_aux_t, vmm_aux3, vmm_aux2);

    // -exp(-x*x)
    h->fmul(vmm_aux, vmm_aux0, vmm_aux0);
    h->ld1r(vmm_aux2, table_val2("sign_mask"));
    h->orr(BReg(vmm_aux.getIdx()), BReg(vmm_aux.getIdx()), BReg(vmm_aux2.getIdx()));
    exp_emitter->emit_code(
            { vmm_aux.getIdx() },
            { vmm_aux_dst.getIdx() },
            aux_vec_idxs,
            aux_gpr_idxs);
    h->ld1r(vmm_aux2, table_val2("sign_mask"));
    // vmm_aux_dst = -exp(-x*x)
    h->orr(BReg(vmm_aux_dst.getIdx()), BReg(vmm_aux_dst.getIdx()), BReg(vmm_aux2.getIdx()));

    // get sign
    h->and_(BReg(vmm_aux.getIdx()), BReg(vmm_src.getIdx()), BReg(vmm_aux2.getIdx()));

    // -exp(-x*x)*t
    h->fmul(vmm_aux_dst, vmm_aux_dst, vmm_aux_t);

    // compute polynomialial r
    h->ld1r(vmm_aux0, table_val2("erf_pol5"));
    h->ld1r(vmm_aux1, table_val2("erf_pol4"));
    h->fmla(vmm_aux1, vmm_aux0, vmm_aux_t);

    h->ld1r(vmm_aux0, table_val2("erf_pol3"));
    h->fmla(vmm_aux0, vmm_aux1, vmm_aux_t);

    h->ld1r(vmm_aux1, table_val2("erf_pol2"));
    h->fmla(vmm_aux1, vmm_aux0, vmm_aux_t);

    h->ld1r(vmm_aux0, table_val2("erf_pol1"));
    h->fmla(vmm_aux0, vmm_aux1, vmm_aux_t);

    // erf = sign * (1 - r * t * exp(-x*x))
    h->ld1r(vmm_aux2, table_val2("one"));
    h->fmla(vmm_aux2, vmm_aux0, vmm_aux_dst);
    h->orr(BReg(vmm_aux2.getIdx()), BReg(vmm_aux.getIdx()), BReg(vmm_aux2.getIdx()));

    // S = 0.5 * s
    h->ld1r(vmm_aux3, table_val2("half"));
    h->fmul(vmm_dst, vmm_src, vmm_aux3);
    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    h->fmla(vmm_dst, vmm_dst, vmm_aux2);
}

void jit_gelu_erf_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("half", float2int(0.5f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("sign_mask", exec_prc_ == ov::element::f32 ? 0x80000000 : 0x8000, true, exec_prc_);

    push_arg_entry_of("gelu_erf_approx_const", float2int(0.327591091, exec_prc_), true, exec_prc_);
    push_arg_entry_of("gelu_erf_one_over_sqrt_two", float2int(0.707106769, exec_prc_), true, exec_prc_);
    push_arg_entry_of("gelu_erf_one_over_sqrt_pi", float2int(0.564189553, exec_prc_), true, exec_prc_);

    push_arg_entry_of("erf_pol1", float2int(0.254829592f, exec_prc_), true, exec_prc_);  // p1
    push_arg_entry_of("erf_pol2", float2int(-0.284496736f, exec_prc_), true, exec_prc_); // p2
    push_arg_entry_of("erf_pol3", float2int(1.421413741f, exec_prc_), true, exec_prc_);  // p3
    push_arg_entry_of("erf_pol4", float2int(-1.453152027f, exec_prc_), true, exec_prc_); // p4
    push_arg_entry_of("erf_pol5", float2int(1.061405429f, exec_prc_), true, exec_prc_);  // p5
}

void jit_gelu_erf_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_gelu_erf_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// GELU_TANH ///
jit_gelu_tanh_emitter::jit_gelu_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                             dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                             const std::shared_ptr<ov::Node>& node)
                                             : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    tanh_emitter = std::make_unique<jit_tanh_emitter>(h, host_isa, node);
}

jit_gelu_tanh_emitter::jit_gelu_tanh_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                             dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                             const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    tanh_emitter = std::make_unique<jit_tanh_emitter>(h, host_isa, exec_prc);
}

size_t jit_gelu_tanh_emitter::get_inputs_count() const { return 1; }

size_t jit_gelu_tanh_emitter::get_aux_vecs_count() const {
    return std::max<size_t>(tanh_emitter->get_aux_vecs_count() + 2, 3);
}

size_t jit_gelu_tanh_emitter::get_aux_gprs_count() const {
    return tanh_emitter->get_aux_gprs_count() + 1;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_gelu_tanh_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);

    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux0(aux_vec_idxs[std::max<size_t>(tanh_emitter->get_aux_vecs_count(), 1)]);
    const TReg vmm_aux2(aux_vec_idxs[std::max<size_t>(tanh_emitter->get_aux_vecs_count() + 1, 2)]);

    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->fmul(vmm_aux0, vmm_src, vmm_src);
    h->ld1r(vmm_aux1, table_val2("gelu_tanh_fitting_const"));
    h->ld1r(vmm_aux2, table_val2("one"));
    h->fmla(vmm_aux2, vmm_aux1, vmm_aux0);
    h->fmul(vmm_aux2, vmm_src, vmm_aux2);
    h->ld1r(vmm_aux1, table_val2("gelu_tanh_sqrt_two_over_pi"));
    h->fmul(vmm_aux0, vmm_aux1, vmm_aux2);

    const bool store_src = vmm_src.getIdx() == vmm_dst.getIdx();
    if (store_src) {
        h->mov(BReg(vmm_aux2.getIdx()), BReg(vmm_src.getIdx()));
    }

    tanh_emitter->emit_code(
            { vmm_aux0.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->ld1r(vmm_aux1, table_val2("one"));
    h->fadd(vmm_dst, vmm_aux1, vmm_dst);
    h->ld1r(vmm_aux1, table_val2("half"));
    h->fmul(vmm_dst, vmm_aux1, vmm_dst);
    h->fmul(vmm_dst, store_src ? vmm_aux2 : vmm_src, vmm_dst);
}

void jit_gelu_tanh_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("half", float2int(0.5f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("gelu_tanh_fitting_const", float2int(0.0447149985, exec_prc_), true, exec_prc_);
    push_arg_entry_of("gelu_tanh_fitting_const_times_three", float2int(0.134145007, exec_prc_), true, exec_prc_);
    push_arg_entry_of("gelu_tanh_sqrt_two_over_pi", float2int(0.797884583, exec_prc_), true, exec_prc_);
}

void jit_gelu_tanh_emitter::emit_data() const {
    jit_emitter::emit_data();
    tanh_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_gelu_tanh_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// HARD_SWISH ///
jit_hswish_emitter::jit_hswish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
}

jit_hswish_emitter::jit_hswish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
}

size_t jit_hswish_emitter::get_inputs_count() const { return 1; }

size_t jit_hswish_emitter::get_aux_vecs_count() const { return 2; }

size_t jit_hswish_emitter::get_aux_gprs_count() const { return 1; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_hswish_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);
    TReg aux0 = TReg(aux_vec_idxs[0]);
    TReg aux1 = TReg(aux_vec_idxs[1]);

    // result = (x * min(max(x + 3, 0), 6)) / 6
    h->ld1r(aux0, table_val2("three"));
    h->fadd(aux0, src, aux0);
    h->ld1r(aux1, table_val2("zero"));
    h->fmaxnm(aux0, aux0, aux1);
    h->ld1r(aux1, table_val2("six"));
    h->fminnm(aux0, aux0, aux1);
    h->fmul(aux0, aux0, src);
    h->ld1r(aux1, table_val2("one_sixth"));
    h->fmul(dst, aux0, aux1);
}

void jit_hswish_emitter::register_table_entries() {
    push_arg_entry_of("zero", float2int(0.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("three", float2int(3.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("six", float2int(6.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("one_sixth", float2int(1.f/6.f, exec_prc_), true, exec_prc_);
}

std::set<std::vector<element::Type>> jit_hswish_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

/// IS_INF ///

jit_is_inf_emitter::jit_is_inf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {

    auto isInf = ov::as_type_ptr<ov::op::v10::IsInf>(node);
    if (isInf == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to ov::op::v10::IsInf");
    }

    const auto& attributes = isInf->get_attributes();
    detect_negative = attributes.detect_negative;
    detect_positive = attributes.detect_positive;

    prepare_table();
}

jit_is_inf_emitter::jit_is_inf_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                       dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                       const bool detect_negative,
                                       const bool detect_positive,
                                       const ov::element::Type exec_prc)
    : jit_emitter(host, host_isa, exec_prc),
      detect_negative{detect_negative},
      detect_positive{detect_positive} {
    prepare_table();
}

size_t jit_is_inf_emitter::get_inputs_count() const {
    return 1;
}

size_t jit_is_inf_emitter::get_aux_vecs_count() const {
    return 1;
}

size_t jit_is_inf_emitter::get_aux_gprs_count() const {
    return 1;
}

std::set<std::vector<element::Type>> jit_is_inf_emitter::get_supported_precisions(
    const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

void jit_is_inf_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                   const std::vector<size_t>& out_vec_idxs) const {
    if (host_isa_ == dnnl::impl::cpu::aarch64::asimd) {
        emit_isa<dnnl::impl::cpu::aarch64::asimd>(in_vec_idxs, out_vec_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("Can't create jit eltwise kernel");
    }
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
void jit_is_inf_emitter::emit_isa(const std::vector<size_t>& in_vec_idxs,
                                  const std::vector<size_t>& out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT(exec_prc_ == ov::element::f32, "unsupported precision: " + exec_prc_.to_string());

    using TReg = typename dnnl::impl::cpu::aarch64::cpu_isa_traits<isa>::TReg;
    const TReg src = TReg(in_vec_idxs[0]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    if (detect_negative || detect_positive) {
        if (detect_positive) {
            if (detect_negative) {
                // If both positive and negative infinity detection is requested
                // calculate the absolute value of 'src'.
                h->fabs(src.s, src.s);
            }
            // Load 'aux' with positive infinity.
            h->ld1r(aux.s, table_val2("inf"));
        } else if (detect_negative) {
            // Load 'aux' with negative infinity.
            h->ld1r(aux.s, table_val2("inf_neg"));
        }
        // Compare elements of 'src' with 'aux'.
        h->fcmeq(dst.s, src.s, aux.s);
        // Sets elements in 'dst' to 1.0 where the comparison was true.
        h->ld1r(aux.s, table_val2("one"));
        h->and_(dst.b16, dst.b16, aux.b16);

    } else {
        // If neither positive nor negative infinity detection is enabled,
        // set 'dst' with zeros (a eor a is 0)
        h->eor(dst.b16, dst.b16, dst.b16);
    }
}

void jit_is_inf_emitter::register_table_entries() {
    // Registers constant values that comply with the IEEE 754 standard.
    push_arg_entry_of("one", 0x3F800000, true, exec_prc_);
    push_arg_entry_of("inf", 0x7F800000, true, exec_prc_);
    push_arg_entry_of("inf_neg", 0xFF800000, true, exec_prc_);
}

/// MAX ///
jit_maximum_emitter::jit_maximum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_maximum_emitter::jit_maximum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_maximum_emitter::get_inputs_count() const { return 2; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_maximum_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src1 = TReg(in_vec_idxs[0]);
    TReg src2 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fmaxnm(dst, src1, src2);
}

std::set<std::vector<element::Type>> jit_maximum_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// MIN ///
jit_minimum_emitter::jit_minimum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_minimum_emitter::jit_minimum_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_minimum_emitter::get_inputs_count() const { return 2; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_minimum_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src1 = TReg(in_vec_idxs[0]);
    TReg src2 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fminnm(dst, src1, src2);
}

std::set<std::vector<element::Type>> jit_minimum_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// MISH ///
jit_mish_emitter::jit_mish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, node);
}

jit_mish_emitter::jit_mish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const ov::element::Type exec_prc) : jit_emitter(host, host_isa, exec_prc) {
    prepare_table();
    exp_emitter = std::make_unique<jit_exp_emitter>(h, host_isa, exec_prc);
}

size_t jit_mish_emitter::get_inputs_count() const { return 1; }

size_t jit_mish_emitter::get_aux_vecs_count() const {
    return std::max<size_t>(exp_emitter->get_aux_vecs_count() + 1, 2);
}

size_t jit_mish_emitter::get_aux_gprs_count() const {
    return exp_emitter->get_aux_gprs_count() + 1;
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_mish_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    // An equation other than mish(x) = x*tanh(srelu(x)) was used
    // to calculate mish, but it should be remembered that it is equivalent
    // equation, it uses the following rule:
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x),
    // hence the equation for mish can take the form:
    // mish(x) = x * ((e^x + 1)^2 - 1)/((e^x + 1)^2 + 1).
    // This option was chosen because computing tanh requires more registers
    // than exp, and also requires more constants to be stored in memory,
    // making the algorithm slower.

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;

    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_aux0(aux_vec_idxs[0]);
    const TReg vmm_aux2(std::max<size_t>(exp_emitter->get_aux_vecs_count(), 1));

    h->ld1r(vmm_aux0, table_val2("fwd_mish_max_x_for_equation_f"));
    h->fminnm(vmm_aux2, vmm_src, vmm_aux0);

    exp_emitter->emit_code(
            { vmm_aux2.getIdx() },
            { vmm_aux2.getIdx() },
            aux_vec_idxs,
            aux_gpr_idxs);

    // (e^x+1)^2
    h->fmov(vmm_aux0, 1.f);
    h->fadd(vmm_aux2, vmm_aux2, vmm_aux0);
    h->fmul(vmm_dst, vmm_aux2, vmm_aux2);

    // save (e^x+1)^2 as it appears in both the denominator and the numerator
    const TReg vmm_aux_src(aux_vec_idxs[1]);
    h->mov(BReg(vmm_aux_src.getIdx()), BReg(vmm_dst.getIdx()));

    // x * ((e^x + 1)^2 - 1) / ((e^x + 1)^2 + 1)
    h->fsub(vmm_aux_src, vmm_aux_src, vmm_aux0);
    h->fadd(vmm_dst, vmm_dst, vmm_aux0);
    h->fdiv(vmm_dst, vmm_aux_src, vmm_dst);
    h->fmul(vmm_dst, vmm_dst, vmm_src);
}

void jit_mish_emitter::register_table_entries() {
    push_arg_entry_of("fwd_mish_max_x_for_equation_f", float2int(44.3614159, exec_prc_), true, exec_prc_);
}

void jit_mish_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_mish_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f32}};
}

/// MOD ///
jit_mod_emitter::jit_mod_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                 dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                 const std::shared_ptr<ov::Node>& node)
    : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
}

jit_mod_emitter::jit_mod_emitter(dnnl::impl::cpu::aarch64::jit_generator *host,
                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                         const ov::element::Type exec_prc): jit_emitter(host, host_isa, exec_prc) {
}

size_t jit_mod_emitter::get_inputs_count() const { return 2; }

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_mod_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;

    TReg divend = TReg(in_vec_idxs[0]);
    TReg divisor = TReg(in_vec_idxs[1]);
    TReg r = TReg(out_vec_idxs[0]);

    h->uni_fdiv(r, divend, divisor);
    h->frintz(r, r);
    h->uni_fmul(r, r, divisor);
    h->uni_fsub(r, divend, r);
}

std::set<std::vector<element::Type>> jit_mod_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_mul_add_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg dst = TReg(out_vec_idxs[0]);

    TReg mul0(in_vec_idxs[0]);
    if (dst.getIdx() == in_vec_idxs[0]) {
        h->mov(BReg(aux_vec_idxs[0]), BReg(in_vec_idxs[0]));
        mul0 = TReg(aux_vec_idxs[0]);
    }

    TReg mul1(in_vec_idxs[1]);
    if (dst.getIdx() == in_vec_idxs[1]) {
        h->mov(BReg(aux_vec_idxs[0]), BReg(in_vec_idxs[1]));
        mul1 = TReg(aux_vec_idxs[0]);
    }

    if (dst.getIdx() != in_vec_idxs[2]) {
        h->mov(BReg(dst.getIdx()), BReg(in_vec_idxs[2]));
    }

    h->fmla(dst, mul0, mul1);
}

std::set<std::vector<element::Type>> jit_mul_add_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16, element::f16}, {element::f32, element::f32, element::f32}};
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_multiply_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fmul(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_multiply_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
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
    push_arg_entry_of("power", float2int(power, exec_prc_), true, exec_prc_);
    push_arg_entry_of("scale", float2int(scale, exec_prc_), true, exec_prc_);
    push_arg_entry_of("shift", float2int(shift, exec_prc_), true, exec_prc_);
}

std::set<std::vector<element::Type>> jit_power_static_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_power_static_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;

    TReg dst = TReg(out_vec_idxs[0]);

    if (power == 0.f) {
        h->fmov(dst, 1.);
        return;
    }

    bool get_from_dst = false;
    const auto src = [&in_vec_idxs, &out_vec_idxs, &get_from_dst]() -> TReg {
        return get_from_dst ? TReg(out_vec_idxs[0]) : TReg(in_vec_idxs[0]);
    };

    TReg aux = TReg(aux_vec_idxs[0]);
    if (scale != 1.f) {
        auto adr = table_val2("scale");
        h->ld1r(aux, adr);
        h->fmul(dst, src(), aux);
        get_from_dst = true;
    }

    if (shift != 0.f) {
        auto adr = table_val2("shift");
        h->ld1r(aux, adr);
        h->fadd(dst, src(), aux);
        get_from_dst = true;
    }

    if (power == 1.f) {
        if (!get_from_dst && (in_vec_idxs[0] != dst.getIdx())) {
            h->mov(BReg(dst.getIdx()), BReg(src().getIdx()));
        }
        return;
    }

    if (std::floor(power) == power && power > 0) {
        h->mov(BReg(aux.getIdx()), BReg(src().getIdx()));
        h->fmov(dst, 1.);

        auto current_power = static_cast<size_t>(power);
        while (current_power > 0) {
            if (current_power & 1) {
                h->fmul(dst, dst, aux);
            }
            if (current_power > 1) {
                h->fmul(aux, aux, aux);
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
        const auto length = exec_prc_ == ov::element::f32 ? 4 : 8;
        for (auto i = 0; i < length; i++) {
            if (exec_prc_ == ov::element::f32) {
                Xbyak_aarch64::VReg4S src2(get_from_dst ? out_vec_idxs[0] : in_vec_idxs[0]);
                h->mov(s0, src2[i]);
                h->ldr(s1, table_val2("power"));
            } else if (exec_prc_ == ov::element::f16) {
                Xbyak_aarch64::VReg8H src2(get_from_dst ? out_vec_idxs[0] : in_vec_idxs[0]);
                Xbyak_aarch64::HReg h0(0);
                h->mov(h0, src2[i]);
                h->fcvt(s0, h0);

                Xbyak_aarch64::HReg h1(1);
                h->ldr(h1, table_val2("power"));
                h->fcvt(s1, h1);
            }

            if (exec_prc_ == ov::element::f32) {
                Xbyak_aarch64::VReg4S src2(get_from_dst ? out_vec_idxs[0] : in_vec_idxs[0]);
                h->mov(s0, src2[i]);
                h->ldr(s1, table_val("power"));
            } else if (exec_prc_ == ov::element::f16) {
                Xbyak_aarch64::VReg8H src2(get_from_dst ? out_vec_idxs[0] : in_vec_idxs[0]);
                Xbyak_aarch64::HReg h0(0);
                h->mov(h0, src2[i]);
                h->fcvt(s0, h0);

                Xbyak_aarch64::HReg h1(1);
                h->ldr(h1, table_val("power"));
                h->fcvt(s1, h1);
            }

            h->str(Xbyak_aarch64::QReg(dst.getIdx()), pre_ptr(h->sp, -16));
            h->str(Xbyak_aarch64::QReg(src().getIdx()), pre_ptr(h->sp, -16));
            h->blr(func_reg);
            h->ldr(Xbyak_aarch64::QReg(src().getIdx()), post_ptr(h->sp, 16));
            h->ldr(Xbyak_aarch64::QReg(dst.getIdx()), post_ptr(h->sp, 16));

            Xbyak_aarch64::WReg w0(0);
            if (exec_prc_ == ov::element::f32) {
                h->fmov(w0, s0);
            } else if (exec_prc_ == ov::element::f16) {
                Xbyak_aarch64::HReg h0(0);
                h->fcvt(h0, s0);
                h->fmov(w0, h0);
            }
            h->mov(dst[i], w0);
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
    return {{element::f16}, {element::f32}};
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_prelu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[0]);
    TReg src2 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->fcmge(dst, src1, 0.0);
    h->fmul(tmp, src1, src2);
    h->bsl(BReg(dst.getIdx()), BReg(src1.getIdx()), BReg(tmp.getIdx()));
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
    return {{element::f16}, {element::f32}};
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_relu_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;

    TReg tmp = TReg(aux_vec_idxs[0]);
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->movi(tmp, 0);
    h->fmaxnm(dst, src, tmp);
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
    return {{element::f16, element::f16, element::f16}, {element::f32, element::f32, element::f32}};
}

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_select_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg src1 = TReg(in_vec_idxs[0]);
    const TReg src2 = TReg(in_vec_idxs[1]);
    const TReg src3 = TReg(in_vec_idxs[2]);
    const TReg dst = TReg(out_vec_idxs[0]);
    const TReg aux = TReg(aux_vec_idxs[0]);

    h->eor(BReg(aux.getIdx()), BReg(aux.getIdx()), BReg(aux.getIdx()));
    h->fcmgt(aux, src1, aux);

    h->bsl(BReg(aux.getIdx()), BReg(src2.getIdx()), BReg(src3.getIdx()));
    h->mov(BReg(dst.getIdx()), BReg(aux.getIdx()));
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_sigmoid_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);

    const TReg vmm_aux0(aux_vec_idxs[exp_emitter->get_aux_vecs_count() + 1]);
    const TReg vmm_mask(aux_vec_idxs[exp_emitter->get_aux_vecs_count()]);

    // To avoid exp(x) overflow happened at x > logf(FLT_MAX), negate positive,
    // compute exp(x), where x <= 0 to get 0 <= exp(x) <= 1 and restore value
    // sign at the end. This is possible due to logistic is symmetric function.
    // IMPORTANT: we use vmm_mask for the mask as exp_compute does not use it.
    // we store the original sign and make x negative
    h->eor(BReg(vmm_aux0.getIdx()), BReg(vmm_aux0.getIdx()), BReg(vmm_aux0.getIdx()));
    h->fcmgt(vmm_mask, vmm_src, vmm_aux0);

    h->ld1r(vmm_aux0, table_val2("sign_mask"));
    h->orr(BReg(vmm_aux0.getIdx()), BReg(vmm_src.getIdx()), BReg(vmm_aux0.getIdx()));

    exp_emitter->emit_code(
            { vmm_aux0.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    const TReg vmm_aux1(aux_vec_idxs[0]);
    const TReg vmm_aux2(aux_vec_idxs[1]);
    // dup exp(x)
    h->mov(BReg(vmm_aux1.getIdx()), BReg(vmm_dst.getIdx()));
    // (exp(x) + 1)
    h->ld1r(vmm_aux0, table_val2("one"));
    h->fadd(vmm_aux1, vmm_aux1, vmm_aux0);
    // y = exp(x) / (exp(x) + 1)
    h->fdiv(vmm_dst, vmm_dst, vmm_aux1);

    // Now we have to apply the "symmetry" based on original sign
    h->ld1r(vmm_aux2, table_val2("one"));
    h->fsub(vmm_aux2, vmm_aux2, vmm_dst);

    h->bsl(BReg(vmm_mask.getIdx()), BReg(vmm_aux2.getIdx()), BReg(vmm_dst.getIdx()));
    h->mov(BReg(vmm_dst.getIdx()), BReg(vmm_mask.getIdx()));
}

void jit_sigmoid_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("sign_mask", exec_prc_ == ov::element::f32 ? 0x80000000 : 0x8000, true, exec_prc_);
}

void jit_sigmoid_emitter::emit_data() const {
    jit_emitter::emit_data();
    exp_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_sigmoid_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_subtract_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src0 = TReg(in_vec_idxs[0]);
    TReg src1 = TReg(in_vec_idxs[1]);
    TReg dst = TReg(out_vec_idxs[0]);

    h->uni_fsub(dst, src0, src1);
}

std::set<std::vector<element::Type>> jit_subtract_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16, element::f16}, {element::f32, element::f32}};
}

/// SWISH ///
jit_swish_emitter::jit_swish_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                     const std::shared_ptr<ov::Node>& node)
        : jit_emitter(host, host_isa, node, get_arithmetic_binary_exec_precision(node)) {
    const auto swish = std::dynamic_pointer_cast<SwishNode>(node);
    if (swish == nullptr) {
        OV_CPU_JIT_EMITTER_THROW("Can't cast to SwishNode");
    }
    beta = static_cast<float>(swish->get_alpha());

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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_swish_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    using BReg = typename cpu_isa_vector_traits<isa, type>::BReg;
    const TReg vmm_src(in_vec_idxs[0]);
    const TReg vmm_dst(out_vec_idxs[0]);
    const TReg vmm_orig_src(aux_vec_idxs[sigmoid_emitter->get_aux_vecs_count()]);
    const TReg vmm_aux(aux_vec_idxs[sigmoid_emitter->get_aux_vecs_count() + 1]);

    h->mov(BReg(vmm_orig_src.getIdx()), BReg(vmm_src.getIdx()));

    // x*beta
    h->ld1r(vmm_aux, table_val2("beta"));
    h->fmul(vmm_aux, vmm_aux, vmm_src);

    // sigmoid(x*beta)
    sigmoid_emitter->emit_code(
            { vmm_aux.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    // x*sigmoid(x*beta)
    h->fmul(vmm_dst, vmm_dst, vmm_orig_src);
}

void jit_swish_emitter::register_table_entries() {
    push_arg_entry_of("beta", float2int(beta, exec_prc_), true, exec_prc_);
}

void jit_swish_emitter::emit_data() const {
    jit_emitter::emit_data();
    sigmoid_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_swish_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
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

template <dnnl::impl::cpu::aarch64::cpu_isa_t isa, typename type>
void jit_tanh_emitter::emit_isa(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs) const {
    OV_CPU_JIT_EMITTER_ASSERT_FP16_FP32(exec_prc_)

    using TReg = typename cpu_isa_vector_traits<isa, type>::TReg;
    TReg src = TReg(in_vec_idxs[0]);
    TReg dst = TReg(out_vec_idxs[0]);

    TReg aux = TReg(aux_vec_idxs.back());

    h->ld1r(aux, table_val2("two"));
    h->uni_fmul(aux, src, aux);

    sigmoid_emitter->emit_code(
            { aux.getIdx() },
            out_vec_idxs,
            aux_vec_idxs,
            aux_gpr_idxs);

    h->ld1r(aux, table_val2("two"));
    h->uni_fmul(dst, aux, dst);
    h->ld1r(aux, table_val2("one"));
    h->uni_fsub(dst, dst, aux);
}

void jit_tanh_emitter::register_table_entries() {
    push_arg_entry_of("one", float2int(1.f, exec_prc_), true, exec_prc_);
    push_arg_entry_of("two", float2int(2.f, exec_prc_), true, exec_prc_);
}

void jit_tanh_emitter::emit_data() const {
    jit_emitter::emit_data();
    sigmoid_emitter->emit_data();
}

std::set<std::vector<element::Type>> jit_tanh_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {{element::f16}, {element::f32}};
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
