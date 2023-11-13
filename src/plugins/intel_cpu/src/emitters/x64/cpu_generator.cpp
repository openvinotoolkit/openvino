// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/snippets_isa.hpp"

#include <string>
#include <iostream>
#include <array>

#include "cpu_generator.hpp"
#include "jit_snippets_emitters.hpp"
#include "jit_eltwise_emitters.hpp"
#include "jit_dnnl_emitters.hpp"
#include "jit_dnnl_ext_emitters.hpp"
#include "jit_conversion_emitters.hpp"

#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"
#include "transformations/snippets/x64/op/fused_mul_add.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"

#include <ngraph/opsets/opset5.hpp>

namespace ov {

#define CREATE_SNIPPETS_EMITTER(e_type) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return std::make_shared<e_type>(h.get(), isa, expr); \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return e_type::get_supported_precisions(n); \
    } \
}

#define CREATE_CPU_EMITTER(e_type) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return std::make_shared<e_type>(h.get(), isa, expr->get_node()); \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return e_type::get_supported_precisions(n); \
    } \
}

class jit_snippet : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator(jit_name()) {
    }

    void generate() override {
    }
};

intel_cpu::CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa)
    : TargetMachine(), h(new jit_snippet()), isa(host_isa) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);
    jitters[snippets::op::Buffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);
    jitters[snippets::op::VectorBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);
    jitters[snippets::op::RankNormalization::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter);
    // jitters[op::v1::Constant::get_type_info_static()] = CREATE_CPU_EMITTER(); // Not supported

    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoadEmitter);
    jitters[snippets::op::LoadReshape::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoadEmitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BroadcastLoadEmitter);
    jitters[intel_cpu::LoadConvertSaturation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoadConvertEmitter);
    jitters[intel_cpu::LoadConvertTruncation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoadConvertEmitter);

    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(StoreEmitter);
    jitters[intel_cpu::StoreConvertSaturation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(StoreConvertEmitter);
    jitters[intel_cpu::StoreConvertTruncation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(StoreConvertEmitter);

    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ScalarEmitter);
    jitters[snippets::op::BroadcastMove::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BroadcastMoveEmitter);
    // jitters[snippets::op::Nop::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(NopEmitter); // Not supported
    // jitters[op::v1::Broadcast::get_type_info_static()] = CREATE_CPU_EMITTER(); // Not supported

    jitters[snippets::op::ConvertTruncation::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_convert_truncation_emitter);
    jitters[snippets::op::ConvertSaturation::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_convert_saturation_emitter);
    // jitters[op::v1::FakeQuantize::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported

    // ternary
    jitters[op::v1::Select::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_select_emitter);
    jitters[intel_cpu::FusedMulAdd::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_mul_add_emitter);

    // binary
    jitters[op::v1::Add::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_add_emitter);
    jitters[op::v1::Divide::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_divide_emitter);
    jitters[op::v1::Equal::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_equal_emitter);
    jitters[op::v1::FloorMod::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_floor_mod_emitter);
    jitters[op::v1::Greater::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_greater_emitter);
    jitters[op::v1::GreaterEqual::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_greater_equal_emitter);
    jitters[op::v1::Less::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_less_emitter);
    jitters[op::v1::LessEqual::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_less_equal_emitter);
    jitters[op::v1::LogicalAnd::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_and_emitter);
    jitters[op::v1::LogicalOr::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_or_emitter);
    jitters[op::v1::LogicalXor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_xor_emitter);
    jitters[op::v1::Maximum::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_maximum_emitter);
    jitters[op::v1::Minimum::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_minimum_emitter);
    jitters[op::v1::Mod::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_mod_emitter);
    jitters[op::v1::Multiply::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_multiply_emitter);
    jitters[op::v1::NotEqual::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_not_equal_emitter);
    jitters[snippets::op::PowerStatic::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_power_static_emitter);
    jitters[op::v1::Power::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_power_dynamic_emitter);
    jitters[op::v0::PRelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_prelu_emitter);
    jitters[op::v0::SquaredDifference::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_squared_difference_emitter);
    jitters[op::v1::Subtract::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_subtract_emitter);
    jitters[op::v0::Xor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_xor_emitter);

    // unary
    jitters[op::v0::Abs::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_abs_emitter);
    // jitters[op::v1::Acos::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    // jitters[op::v1::Asin::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    // jitters[op::v1::Atan::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[op::v0::Ceiling::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_ceiling_emitter);
    jitters[op::v0::Clamp::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_clamp_emitter);
    // jitters[op::v1::Cos::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    // jitters[op::v1::Cosh::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[op::v0::Elu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_elu_emitter);
    jitters[op::v0::Erf::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_erf_emitter);
    jitters[op::v0::Exp::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_exp_emitter);
    jitters[op::v0::Floor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_floor_emitter);
    jitters[ngraph::opset5::Round::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_round_emitter);
    // jitters[op::v1::Log::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[op::v1::LogicalNot::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_not_emitter);
    jitters[op::v0::Negative::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_negative_emitter);
    jitters[op::v0::Relu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_relu_emitter);
    // jitters[op::v1::Sign::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[op::v0::Sigmoid::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_sigmoid_emitter);
    // jitters[op::v1::Sin::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    // jitters[op::v1::Sinh::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[op::v0::Sqrt::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_sqrt_emitter);
    // jitters[op::v1::Tan::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[op::v0::Tanh::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_tanh_emitter);

    jitters[intel_cpu::SwishNode::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_swish_emitter);
    jitters[ngraph::op::v4::HSwish::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_hswish_emitter);
    // jitters[op::v1::HardSigmoid::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    // jitters[op::v1::Selu::get_type_info_static()] = CREATE_CPU_EMITTER(); // not supported
    jitters[ngraph::op::v0::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_gelu_v0_emitter);
    jitters[ngraph::op::v7::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_gelu_v7_emitter);
    jitters[snippets::op::Fill::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(FillEmitter);

    jitters[snippets::op::HorizonMax::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(HorizonEmitter);
    jitters[snippets::op::HorizonSum::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(HorizonEmitter);

    jitters[snippets::op::Kernel::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(KernelEmitter);
    jitters[snippets::op::LoopBegin::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoopBeginEmitter);
    jitters[snippets::op::LoopEnd::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(LoopEndEmitter);
    jitters[intel_cpu::BrgemmCPU::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BrgemmEmitter);
    jitters[intel_cpu::BrgemmCopyB::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BrgemmCopyBEmitter);
}

size_t intel_cpu::CPUTargetMachine::get_lanes() const {
    switch (isa) {
        case dnnl::impl::cpu::x64::avx2 : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx2>::vlen / sizeof(float);
        case dnnl::impl::cpu::x64::sse41 : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / sizeof(float);
        case dnnl::impl::cpu::x64::avx512_core : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx512_core>::vlen / sizeof(float);
        default : OPENVINO_THROW("unknown isa ", isa);
    }
}

dnnl::impl::cpu::x64::cpu_isa_t intel_cpu::CPUTargetMachine::get_isa() const {
    return isa;
}

bool intel_cpu::CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::x64::mayiuse(isa);
}

snippets::CompiledSnippetPtr intel_cpu::CPUTargetMachine::get_snippet() {
    if (h->create_kernel() != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to create jit_kernel in get_snippet()");
    }
    const auto& result = std::make_shared<CompiledSnippetCPU>(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h.reset(new jit_snippet());
    return result;
}

intel_cpu::CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h) : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* intel_cpu::CompiledSnippetCPU::get_code() const {
    return h_compiled->jit_ker();
}

size_t intel_cpu::CompiledSnippetCPU::get_code_size() const {
    return h_compiled->getSize();
}

bool intel_cpu::CompiledSnippetCPU::empty() const {
    return get_code_size() == 0;
}

intel_cpu::CPUGenerator::CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa_) : Generator(std::make_shared<CPUTargetMachine>(isa_)) {
}

std::shared_ptr<snippets::Generator> intel_cpu::CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    OPENVINO_ASSERT(cpu_target_machine, "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine->get_isa());
}

snippets::Generator::opRegType intel_cpu::CPUGenerator::get_specific_op_reg_type(const std::shared_ptr<ov::Node>& op) const {
    if (std::dynamic_pointer_cast<intel_cpu::BrgemmCPU>(op) ||
        std::dynamic_pointer_cast<intel_cpu::BrgemmCopyB>(op))
        return gpr2gpr;
    else if (
        std::dynamic_pointer_cast<intel_cpu::FusedMulAdd>(op) ||
        std::dynamic_pointer_cast<intel_cpu::SwishNode>(op))
        return vec2vec;
    else
        OPENVINO_THROW("Register type of the operation " + std::string(op->get_type_name()) + " isn't determined!");
}
bool intel_cpu::CPUGenerator::uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& e) const {
    return std::dynamic_pointer_cast<intel_cpu::BrgemmEmitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::BrgemmCopyBEmitter>(e);
}
} // namespace ov