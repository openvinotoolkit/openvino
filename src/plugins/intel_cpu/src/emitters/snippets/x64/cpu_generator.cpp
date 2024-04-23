// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"

#include "emitters/snippets/x64/jit_brgemm_copy_b_emitter.hpp"
#include "emitters/snippets/x64/jit_brgemm_emitter.hpp"
#include "emitters/snippets/x64/jit_memory_emitters.hpp"
#include "emitters/snippets/x64/jit_kernel_emitter.hpp"
#include "emitters/snippets/x64/jit_loop_emitters.hpp"
#include "emitters/snippets/x64/jit_snippets_emitters.hpp"
#include "emitters/snippets/x64/jit_fill_emitter.hpp"
#include "emitters/snippets/x64/jit_horizon_emitter.hpp"
#include "emitters/plugin/x64/jit_eltwise_emitters.hpp"
#include "emitters/plugin/x64/jit_dnnl_emitters.hpp"
#include "emitters/plugin/x64/jit_dnnl_ext_emitters.hpp"
#include "emitters/plugin/x64/jit_conversion_emitters.hpp"

#include "transformations/snippets/x64/op/load_convert.hpp"
#include "transformations/snippets/x64/op/store_convert.hpp"
#include "transformations/snippets/x64/op/fused_mul_add.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"

#include <openvino/opsets/opset5.hpp>

#ifdef SNIPPETS_DEBUG_CAPS
#include "emitters/snippets/x64/jit_perf_count_chrono_emitters.hpp"
#include "emitters/snippets/x64/jit_perf_count_rdtsc_emitters.hpp"
#include "transformations/snippets/x64/op/perf_count_rdtsc.hpp"
#include "emitters/snippets/x64/jit_debug_emitter.hpp"
#include "emitters/snippets/x64/jit_segfault_detector_emitter.hpp"
#include "emitters/snippets/x64/verbose.hpp"
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
#include "transformations/tpp/x64/op/brgemm.hpp"
#include "transformations/tpp/x64/op/eltwise.hpp"
#include "transformations/tpp/x64/op/reduce.hpp"
#include "transformations/tpp/x64/op/modifiers.hpp"
#include "transformations/tpp/x64/op/scalar.hpp"
#include "emitters/tpp/x64/jit_eltwise_emitters.hpp"
#include "emitters/tpp/x64/jit_brgemm_emitter.hpp"
#include "emitters/tpp/x64/jit_scalar_emitter.hpp"
#include "emitters/tpp/x64/jit_debug_emitter.hpp"
// Note: for reference implementations
#include <cmath>
#endif

namespace ov {

#ifdef SNIPPETS_DEBUG_CAPS
static bool is_load_emitter(const intel_cpu::jit_emitter *emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::jit_load_memory_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_load_broadcast_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_load_convert_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_store_emitter(const intel_cpu::jit_emitter *emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::jit_store_memory_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_store_convert_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_segfault_detector_emitter(const intel_cpu::jit_emitter *emitter) {
    // default active for typical tensor memory access emitters
    bool ret = false;
    ret = is_load_emitter(emitter) ||
        is_store_emitter(emitter) ||
        dynamic_cast<const intel_cpu::jit_brgemm_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_brgemm_copy_b_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::jit_kernel_emitter*>(emitter);
    return ret;
    // use below code to active all emitters for extend usage
    // return !dynamic_cast<const jit_nop_emitter*>(emitter);
}

#define CREATE_SNIPPETS_EMITTER(e_type) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        auto emitter = std::make_shared<e_type>(h.get(), isa, expr); \
        if (debug_config.enable_segfault_detector && is_segfault_detector_emitter(emitter.get())) { \
            auto segfault_emitter = std::make_shared<jit_uni_segfault_detector_emitter>(h.get(), isa, emitter.get(), \
                is_load_emitter(emitter.get()), is_store_emitter(emitter.get()), expr->get_node()->get_friendly_name()); \
            return std::make_shared<jit_debug_emitter>(emitter, segfault_emitter, jit_debug_emitter::EmissionLocation::preamble); \
        } else { \
            return emitter; \
        } \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return e_type::get_supported_precisions(n); \
    } \
}
#else
#define CREATE_SNIPPETS_EMITTER(e_type, ...) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__); \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return e_type::get_supported_precisions(n); \
    } \
}
#endif

#define CREATE_DEBUG_TPP_EMITTER(e_type) { \
    [this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return std::make_shared<DebugTppEmitter>(expr, std::make_shared<e_type>(h.get(), isa, expr)); \
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

#define CREATE_UNDEFINED_EMITTER(supported_precisions) { \
    [](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
        return nullptr; \
    }, \
    [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> { \
        return supported_precisions; \
    } \
}

class jit_snippet : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator(jit_name()) {}

    void generate() override {}
};

intel_cpu::CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>()), h(new jit_snippet()), isa(host_isa) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::IntermediateMemoryBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::NewMemoryBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::VectorBuffer::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::RankNormalization::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);
    jitters[snippets::op::Reshape::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_nop_emitter);

    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[snippets::op::LoadReshape::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_memory_emitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_broadcast_emitter);
    jitters[intel_cpu::LoadConvertSaturation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_convert_emitter);
    jitters[intel_cpu::LoadConvertTruncation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_load_convert_emitter);

    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_memory_emitter);
    jitters[intel_cpu::StoreConvertSaturation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_convert_emitter);
    jitters[intel_cpu::StoreConvertTruncation::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_store_convert_emitter);

    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_scalar_emitter);
    jitters[snippets::op::BroadcastMove::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_broadcast_move_emitter);

    jitters[snippets::op::ConvertTruncation::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_convert_truncation_emitter);
    jitters[snippets::op::ConvertSaturation::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_convert_saturation_emitter);

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
    jitters[ov::op::v0::Abs::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_abs_emitter);
    jitters[ov::op::v0::Ceiling::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_ceiling_emitter);
    jitters[ov::op::v0::Clamp::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_clamp_emitter);
    jitters[ov::op::v0::Elu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_elu_emitter);
    jitters[ov::op::v0::Erf::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_erf_emitter);
    jitters[ov::op::v0::Exp::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_exp_emitter);
    jitters[ov::op::v0::Floor::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_floor_emitter);
    jitters[ov::opset5::Round::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_round_emitter);
    jitters[ov::op::v1::LogicalNot::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_logical_not_emitter);
    jitters[ov::op::v0::Negative::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_negative_emitter);
    jitters[ov::op::v0::Relu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_relu_emitter);
    jitters[ov::op::v0::Sigmoid::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_sigmoid_emitter);
    jitters[ov::op::v0::Sqrt::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_sqrt_emitter);
    jitters[ov::op::v0::Tanh::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_tanh_emitter);

    jitters[ov::intel_cpu::SwishNode::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::jit_swish_emitter);
    jitters[ov::op::v4::HSwish::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_hswish_emitter);
    jitters[ov::op::v0::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_gelu_v0_emitter);
    jitters[ov::op::v7::Gelu::get_type_info_static()] = CREATE_CPU_EMITTER(intel_cpu::jit_gelu_v7_emitter);
    jitters[snippets::op::Fill::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_fill_emitter);

    jitters[snippets::op::HorizonMax::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_horizon_emitter);
    jitters[snippets::op::HorizonSum::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_horizon_emitter);

    jitters[snippets::op::KernelStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_kernel_static_emitter);
    jitters[snippets::op::KernelDynamic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_kernel_dynamic_emitter);
    jitters[snippets::op::LoopBeginStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_loop_begin_static_emitter);
    jitters[snippets::op::LoopBeginDynamic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_loop_begin_dynamic_emitter);
    jitters[snippets::op::LoopEndStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_loop_end_static_emitter);
    jitters[snippets::op::LoopEndDynamic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_loop_end_dynamic_emitter);
    jitters[intel_cpu::BrgemmCPU::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_brgemm_emitter);
    jitters[intel_cpu::BrgemmCopyB::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(intel_cpu::jit_brgemm_copy_b_emitter);
    jitters[snippets::op::ReduceMax::get_type_info_static()] = CREATE_UNDEFINED_EMITTER({{ov::element::f32}});
    jitters[snippets::op::ReduceSum::get_type_info_static()] = CREATE_UNDEFINED_EMITTER({{ov::element::f32}});

#ifdef SNIPPETS_DEBUG_CAPS
    jitters[snippets::op::PerfCountBegin::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::jit_perf_count_chrono_start_emitter);
    jitters[snippets::op::PerfCountEnd::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::jit_perf_count_chrono_end_emitter);
    jitters[ov::intel_cpu::PerfCountRdtscBegin::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::jit_perf_count_rdtsc_start_emitter);
    jitters[ov::intel_cpu::PerfCountRdtscEnd::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::jit_perf_count_rdtsc_end_emitter);
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
    jitters[intel_cpu::tpp::op::BrgemmTPP::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BrgemmTppEmitter);
    jitters[intel_cpu::tpp::op::Add::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Subtract::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Multiply::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Divide::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(BinaryEltwiseTppEmitter);

    jitters[intel_cpu::tpp::op::Exp::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    // Note: you can register Debug emitter for Unary/Binary operations as shown below:
    // jitters[intel_cpu::tpp::op::Add::get_type_info_static()] = CREATE_DEBUG_TPP_EMITTER(UnaryEltwiseTppEmitter);
    //
    // Note: you can register Reference emitter for Unary operations using std::function or lambda function as shown below:
    // jitters[intel_cpu::tpp::op::Exp::get_type_info_static()] =
    //        CREATE_SNIPPETS_EMITTER(ReferenceUnaryEltwiseTppEmitter, static_cast<float(*)(float)>(std::exp));
    // jitters[intel_cpu::tpp::op::Reciprocal::get_type_info_static()] =
    //         CREATE_SNIPPETS_EMITTER(ReferenceUnaryEltwiseTppEmitter, [](float x){ return 1.f/x; });
    jitters[intel_cpu::tpp::op::Reciprocal::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);

    jitters[intel_cpu::tpp::op::Relu::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::Square::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::SquareRoot::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(UnaryEltwiseTppEmitter);
    jitters[intel_cpu::tpp::op::ReduceMax::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ReduceTppEmitter);
    jitters[intel_cpu::tpp::op::ReduceSum::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ReduceTppEmitter);
    jitters[intel_cpu::tpp::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(ScalarTppEmitter);
#endif
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

ov::snippets::RegType intel_cpu::CPUGenerator::get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const {
    const auto op = out.get_node_shared_ptr();
    if (std::dynamic_pointer_cast<intel_cpu::BrgemmCPU>(op) ||
#ifdef SNIPPETS_LIBXSMM_TPP
        std::dynamic_pointer_cast<intel_cpu::tpp::modifier::TensorProcessingPrimitive>(op) ||
        std::dynamic_pointer_cast<intel_cpu::tpp::op::Scalar>(op) ||
#endif
        std::dynamic_pointer_cast<intel_cpu::BrgemmCopyB>(op))
        return ov::snippets::RegType::gpr;
    else if (
        std::dynamic_pointer_cast<intel_cpu::FusedMulAdd>(op) ||
        std::dynamic_pointer_cast<intel_cpu::SwishNode>(op))
        return ov::snippets::RegType::vec;
    else
       return ov::snippets::RegType::undefined;
}

bool intel_cpu::CPUGenerator::uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& e) const {
    bool need = std::dynamic_pointer_cast<intel_cpu::jit_brgemm_emitter>(e) ||
                std::dynamic_pointer_cast<intel_cpu::jit_brgemm_copy_b_emitter>(e);
#ifdef SNIPPETS_DEBUG_CAPS
    const auto cpu_target_machine = std::dynamic_pointer_cast<intel_cpu::CPUTargetMachine>(target);
    need = need || (cpu_target_machine && cpu_target_machine->debug_config.enable_segfault_detector) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_chrono_start_emitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_chrono_end_emitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_rdtsc_start_emitter>(e) ||
           std::dynamic_pointer_cast<intel_cpu::jit_perf_count_rdtsc_end_emitter>(e);
#endif
#ifdef SNIPPETS_LIBXSMM_TPP
    need |= std::dynamic_pointer_cast<intel_cpu::ReferenceUnaryEltwiseTppEmitter>(e) ||
            std::dynamic_pointer_cast<intel_cpu::DebugTppEmitter>(e);
#endif
    return need;
}

intel_cpu::CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {}

bool intel_cpu::CPURuntimeConfigurator::is_update_needed(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    if (m_latest_input_shapes.empty())
        return true;
    for (size_t i = 0; i < m_latest_input_shapes.size(); ++i)
        if (m_latest_input_shapes[i] != *m_io_shapes[i])
            return true;
    return false;
}

void intel_cpu::CPURuntimeConfigurator::update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    const auto& cpu_config = std::dynamic_pointer_cast<CPURuntimeConfig>(m_config);
    OPENVINO_ASSERT(cpu_config, "CPURuntimeConfigurator expects CPURuntimeConfig");

    if (linear_ir->is_dynamic()) {
        update_linear_ir_state(linear_ir);
        update_loop_args(linear_ir, cpu_config);
    }

    if (m_io_num == 0) {
        init_data_info(linear_ir);
        cpu_config->tensor_rank = std::max(linear_ir->get_master_shape().size(), rank6D);
    }

    update_data_offsets(cpu_config);
    update_parallel_domain(linear_ir, cpu_config);

    update_latest_shapes();
}

void intel_cpu::CPURuntimeConfigurator::init_data_info(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    const auto& io_exprs = linear_ir->get_IO_ops();
    m_io_num = io_exprs.size();
    m_io_shapes.resize(m_io_num);
    m_io_layouts.resize(m_io_num);
    m_io_data_sizes.resize(m_io_num);
    m_latest_input_shapes.resize(m_io_num);
    m_in_num = 0;

    size_t idx = 0;
    for (const auto& expr : io_exprs) {
        switch (expr->get_type()) {
            case ov::snippets::lowered::IOExpression::io_type::INPUT: {
                // input->shape changing ops->load
                const auto& shape_infer_seq = ov::snippets::utils::get_first_child_shape_infer_expr_seq(expr);
                const auto& mem_desc_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
                auto consumer_inputs = mem_desc_expr->get_output_port_connector(0)->get_consumers();
                for (const auto& child_input : consumer_inputs) {
                    const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(child_input.get_expr()->get_node());
                    if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                        m_io_shapes[idx] = child_input.get_descriptor_ptr()->get_shape_ptr();
                        m_io_layouts[idx] = child_input.get_descriptor_ptr()->get_layout();
                        break;
                    }
                }
                m_io_data_sizes[idx] = mem_desc_expr->get_node()->get_output_element_type(0).size();
                m_in_num++;
                break;
            }
            case ov::snippets::lowered::IOExpression::io_type::OUTPUT: {
                const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(expr);
                const auto& mem_desc_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
                const auto& parent_output = mem_desc_expr->get_input_port_connector(0)->get_source();
                m_io_shapes[idx] = parent_output.get_descriptor_ptr()->get_shape_ptr();
                m_io_layouts[idx] = parent_output.get_descriptor_ptr()->get_layout();
                m_io_data_sizes[idx] = expr->get_node()->get_input_element_type(0).size();
                break;
            } default : {
                OPENVINO_THROW("Detected unsupported io_type");
            }
        }
        OPENVINO_ASSERT(m_io_shapes[idx], "IO Shape is missed!");
        OPENVINO_ASSERT(m_io_shapes[idx]->size() == m_io_layouts[idx].size(), "Incompatible ranks of shape and layout!");
        idx++;
    }
}


void intel_cpu::CPURuntimeConfigurator::update_data_offsets(const std::shared_ptr<CPURuntimeConfig>& cpu_config) const {
    cpu_config->io_data_offsets.resize(m_io_num);
    for (size_t i = 0; i < m_io_num; ++i) {
        // offsets represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    offsets: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    offsets: s1*s3, s3,       0,  1
        const auto& shape = *m_io_shapes[i];
        if (shape == m_latest_input_shapes[i])
            continue;

        const auto& layout = m_io_layouts[i];
        auto& offsets = cpu_config->io_data_offsets[i];

        offsets.resize(cpu_config->tensor_rank);
        std::fill(offsets.begin(), offsets.end(), 0);
        if (ov::snippets::utils::is_dynamic_vdims(shape))
            return;

        size_t dim_step = m_io_data_sizes[i];
        offsets[offsets.size() - 1] = dim_step;

        OPENVINO_ASSERT(cpu_config->tensor_rank >= shape.size(), "Incorrect tensor rank!");
        const auto idx_stride = cpu_config->tensor_rank - shape.size();
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
            dim_step *= shape[i + 1];
            offsets[i + idx_stride] = shape[i] != 1 ? dim_step : 0;
        }
        if (!layout.empty()) {
            std::vector<size_t> reordered_offsets(offsets.size());
            const auto is_input = i < m_in_num;
            for (size_t i = 0; i < layout.size(); i++) {
                const auto& src_idx = is_input ? layout[i] : i;
                const auto& dst_idx = is_input ? i : layout[i];
                reordered_offsets[idx_stride + dst_idx] = offsets[idx_stride + src_idx];
            }
            offsets = std::move(reordered_offsets);
        }
    }
}

void intel_cpu::CPURuntimeConfigurator::update_loop_args(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                                         const std::shared_ptr<CPURuntimeConfig>& cpu_config) const {
    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    cpu_config->loop_args.resize(loop_map.size());
    for (const auto& loop : loop_map) {
        const auto& idx = loop.first;
        const auto& loop_info = std::dynamic_pointer_cast<ov::snippets::lowered::ExpandedLoopInfo>(loop.second);
        OPENVINO_ASSERT(loop_info, "CPURuntimeConfigurator expects ExpandedLoopInfo in loop manager");

        const auto& increment = loop_info->get_increment();
        const auto& data_sizes = loop_info->get_dense_data_sizes();

        auto& loop_arg = cpu_config->loop_args[idx];
        loop_arg = jit_snippets_call_args::loop_args_t(loop_info->get_work_amount(), loop_info->get_dense_ptr_increments(),
                                                       loop_info->get_dense_finalization_offsets());
        for (int64_t i = 0; i < loop_arg.m_num_data_ptrs; ++i) {
            loop_arg.m_ptr_increments[i] *= (increment * data_sizes[i]);
            loop_arg.m_finalization_offsets[i] *= data_sizes[i];
        }
    }
}

void intel_cpu::CPURuntimeConfigurator::update_parallel_domain(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                                               const std::shared_ptr<CPURuntimeConfig>& cpu_config) const {
    cpu_config->parallel_domain.resize(cpu_config->tensor_rank, 1);
    const auto parallel_exec_domain = linear_ir->get_parallel_domain();
    std::copy(parallel_exec_domain.cbegin(), parallel_exec_domain.cend(),
              cpu_config->parallel_domain.begin() + (cpu_config->tensor_rank - parallel_exec_domain.size()));
}

void intel_cpu::CPURuntimeConfigurator::update_latest_shapes() {
    m_latest_input_shapes.resize(m_in_num);
    for (size_t i = 0; i < m_in_num; ++i) {
        m_latest_input_shapes[i] = *m_io_shapes[i];
    }
}

} // namespace ov
