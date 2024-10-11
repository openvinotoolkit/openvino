// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/snippets_isa.hpp"
#include "cpu_generator.hpp"
#include "jit_snippets_emitters.hpp"
#include "emitters/utils.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/plugin/aarch64/jit_eltwise_emitters.hpp"
#include "emitters/plugin/aarch64/jit_conversion_emitters.hpp"
#include "emitters/snippets/aarch64/jit_kernel_emitter.hpp"
#include "emitters/snippets/aarch64/jit_loop_emitters.hpp"
#include "emitters/snippets/aarch64/jit_memory_emitters.hpp"
#include "emitters/snippets/aarch64/jit_fill_emitter.hpp"

#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"

#include "openvino/opsets/opset13.hpp"

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

#define CREATE_GELU_V7_EMITTER(e_type_erf, e_type_tanh) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        const auto& n = expr->get_node(); \
        const auto& gelu = std::dynamic_pointer_cast<ov::op::v7::Gelu>(n); \
        if (gelu == nullptr) { \
            OPENVINO_THROW("Can't cast to ov::op::v7::Gelu"); \
        } \
        const auto approximationMode = gelu->get_approximation_mode(); \
        if (approximationMode == ov::op::GeluApproximationMode::ERF) { \
            return std::make_shared<e_type_erf>(h.get(), isa, n); \
        } else if (approximationMode == ov::op::GeluApproximationMode::TANH) { \
            return std::make_shared<e_type_tanh>(h.get(), isa, n); \
        } else { \
            OPENVINO_THROW("Unsupported Gelu approximation mode"); \
        } \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        const auto& gelu = std::dynamic_pointer_cast<ov::op::v7::Gelu>(n); \
        if (gelu == nullptr) { \
            OPENVINO_THROW("Can't cast to ov::op::v7::Gelu"); \
        } \
        const auto approximationMode = gelu->get_approximation_mode(); \
        if (approximationMode == ov::op::GeluApproximationMode::ERF) { \
            return e_type_erf::get_supported_precisions(n); \
        } else if (approximationMode == ov::op::GeluApproximationMode::TANH) { \
            return e_type_tanh::get_supported_precisions(n); \
        } else { \
            OPENVINO_THROW("Unsupported Gelu approximation mode"); \
        } \
    } \
}

class jit_snippet : public dnnl::impl::cpu::aarch64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    virtual ~jit_snippet() = default;

    jit_snippet() : jit_generator() {}

    void generate() override {}
};

namespace intel_cpu {
namespace aarch64 {

CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h) : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was nopt compiled");
}

const uint8_t* CompiledSnippetCPU::get_code() const {
    return h_compiled->jit_ker();
}

size_t CompiledSnippetCPU::get_code_size() const {
    return h_compiled->getSize();
}

bool CompiledSnippetCPU::empty() const {
    return get_code_size() == 0;
}

CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::aarch64::cpu_isa_t host_isa)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>()), h(new jit_snippet()), isa(host_isa) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::VectorBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::RankNormalization::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::BroadcastMove::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_broadcast_move_emitter);
    jitters[snippets::op::ConvertTruncation::get_type_info_static()] = CREATE_CPU_EMITTER(jit_convert_truncation_emitter);
    jitters[snippets::op::ConvertSaturation::get_type_info_static()] = CREATE_CPU_EMITTER(jit_convert_saturation_emitter);

    // memory access
    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_memory_emitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_broadcast_emitter);
    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_store_memory_emitter);

    // ternary
    jitters[intel_cpu::FusedMulAdd::get_type_info_static()] = CREATE_CPU_EMITTER(jit_mul_add_emitter);

    // binary
    jitters[op::v1::Add::get_type_info_static()] = CREATE_CPU_EMITTER(jit_add_emitter);
    jitters[op::v1::Divide::get_type_info_static()] = CREATE_CPU_EMITTER(jit_divide_emitter);
    jitters[op::v1::Maximum::get_type_info_static()] = CREATE_CPU_EMITTER(jit_maximum_emitter);
    jitters[op::v1::Minimum::get_type_info_static()] = CREATE_CPU_EMITTER(jit_minimum_emitter);
    jitters[op::v1::Mod::get_type_info_static()] = CREATE_CPU_EMITTER(jit_mod_emitter);
    jitters[op::v1::Multiply::get_type_info_static()] = CREATE_CPU_EMITTER(jit_multiply_emitter);
    jitters[op::v1::Subtract::get_type_info_static()] = CREATE_CPU_EMITTER(jit_subtract_emitter);

    // unary
    jitters[ov::op::v0::Abs::get_type_info_static()] = CREATE_CPU_EMITTER(jit_abs_emitter);
    jitters[ov::op::v0::Clamp::get_type_info_static()] = CREATE_CPU_EMITTER(jit_clamp_emitter);
    jitters[ov::op::v0::Elu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_elu_emitter);
    jitters[ov::op::v0::Exp::get_type_info_static()] = CREATE_CPU_EMITTER(jit_exp_emitter);
    jitters[ov::op::v0::Floor::get_type_info_static()] = CREATE_CPU_EMITTER(jit_floor_emitter);
    jitters[ov::op::v0::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_gelu_erf_emitter);
    jitters[ov::op::v7::Gelu::get_type_info_static()] = CREATE_GELU_V7_EMITTER(jit_gelu_erf_emitter, jit_gelu_tanh_emitter);
    jitters[ov::op::v4::HSwish::get_type_info_static()] = CREATE_CPU_EMITTER(jit_hswish_emitter);
    jitters[ov::op::v4::Mish::get_type_info_static()] = CREATE_CPU_EMITTER(jit_mish_emitter);
    jitters[ov::op::v0::Relu::get_type_info_static()] = CREATE_CPU_EMITTER(jit_relu_emitter);
    jitters[ov::op::v0::Sigmoid::get_type_info_static()] = CREATE_CPU_EMITTER(jit_sigmoid_emitter);
    jitters[ov::intel_cpu::SwishNode::get_type_info_static()] = CREATE_CPU_EMITTER(jit_swish_emitter);
    jitters[ov::op::v0::Tanh::get_type_info_static()] = CREATE_CPU_EMITTER(jit_tanh_emitter);

    // control flow
    jitters[snippets::op::KernelStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_static_emitter);
    jitters[snippets::op::KernelDynamic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_dynamic_emitter);
    jitters[snippets::op::LoopBegin::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_loop_begin_emitter);
    jitters[snippets::op::LoopEnd::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_loop_end_emitter);

    // others
    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_scalar_emitter);
    jitters[snippets::op::Fill::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_fill_emitter);
}

std::shared_ptr<snippets::TargetMachine> CPUTargetMachine::clone() const {
    const auto cloned = std::make_shared<CPUTargetMachine>(isa);
    cloned->configurator = std::make_shared<ov::snippets::RuntimeConfigurator>(*configurator);
    return cloned;
}

bool CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
}

snippets::CompiledSnippetPtr CPUTargetMachine::get_snippet() {
    OPENVINO_ASSERT(h->create_kernel() == dnnl::impl::status::success, "Failed to create jit_kernel in get_snippet()");

    const auto& result = std::make_shared<CompiledSnippetCPU>(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h.reset(new jit_snippet());
    return result;
}

size_t CPUTargetMachine::get_lanes() const {
    switch (isa) {
        case dnnl::impl::cpu::aarch64::asimd : return dnnl::impl::cpu::aarch64::cpu_isa_traits<dnnl::impl::cpu::aarch64::asimd>::vlen / sizeof(float);
        default : OPENVINO_THROW("unknown isa ", isa);
    }
}

// TODO [139932]: Support separate vec_count and gpr_count
size_t CPUTargetMachine::get_reg_count() const {
    return 32;
}

dnnl::impl::cpu::aarch64::cpu_isa_t CPUTargetMachine::get_isa() const {
    return isa;
}

CPUGenerator::CPUGenerator(dnnl::impl::cpu::aarch64::cpu_isa_t isa_) : Generator(std::make_shared<CPUTargetMachine>(isa_)) {}

std::shared_ptr<snippets::Generator> CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    OPENVINO_ASSERT(cpu_target_machine, "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine->get_isa());
}

ov::snippets::RegType CPUGenerator::get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const {
    const auto op = out.get_node_shared_ptr();
    if (std::dynamic_pointer_cast<intel_cpu::FusedMulAdd>(op) ||
        std::dynamic_pointer_cast<intel_cpu::SwishNode>(op))
        return ov::snippets::RegType::vec;
    else
        return ov::snippets::RegType::undefined;
}

bool CPUGenerator::uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& e) const {
    return false;
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
