// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/riscv64/jit_eltwise_emitters.hpp"
#include "emitters/snippets/common/emitter_factory.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "jit_kernel_emitter.hpp"
#include "jit_loop_emitters.hpp"
#include "jit_memory_emitters.hpp"
#include "jit_snippets_emitters.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tanh.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/target_machine.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"

#ifdef SNIPPETS_DEBUG_CAPS
#    include "emitters/snippets/riscv64/jit_debug_emitter.hpp"
#    include "emitters/snippets/riscv64/jit_segfault_detector_emitter.hpp"
#    include "emitters/snippets/riscv64/verbose.hpp"
#endif

namespace ov {

#ifdef SNIPPETS_DEBUG_CAPS
static bool is_load_emitter(const intel_cpu::riscv64::jit_emitter* emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::riscv64::jit_load_memory_emitter*>(emitter) ||
        dynamic_cast<const intel_cpu::riscv64::jit_load_broadcast_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_store_emitter(const intel_cpu::riscv64::jit_emitter* emitter) {
    bool ret = false;
    if (dynamic_cast<const intel_cpu::riscv64::jit_store_memory_emitter*>(emitter)) {
        return true;
    }
    return ret;
}

static bool is_segfault_detector_emitter(const intel_cpu::riscv64::jit_emitter* emitter) {
    bool ret = false;
    ret = is_load_emitter(emitter) || is_store_emitter(emitter) ||
          (dynamic_cast<const intel_cpu::riscv64::jit_kernel_emitter*>(emitter) != nullptr);
    return ret;
}

#endif

class jit_snippet : public ov::intel_cpu::riscv64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() override = default;

    // Xbyak_riscv uses a fixed-size code buffer; the default limit can be too small for complex snippets
    // (especially with debug instrumentation enabled), so allocate a larger buffer to avoid JIT failures.
    static constexpr size_t max_code_size_bytes = 64 * 1024;

    jit_snippet() : jit_generator_t(max_code_size_bytes) {}

    void generate() override {}

    void set_code_section_address(const uint8_t* addr) {
        code_section_address = addr;
    }

protected:
    const uint8_t* getCodeAddress() const override {
        return code_section_address ? code_section_address : CodeGenerator::getCode();
    }

private:
    const uint8_t* code_section_address = nullptr;
};

namespace intel_cpu::riscv64 {

CPUTargetMachine::CPUTargetMachine(ov::intel_cpu::riscv64::cpu_isa_t host_isa, ov::intel_cpu::MultiCacheWeakPtr cache)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>(cache)),
      h(new jit_snippet()),
      isa(host_isa),
      compiled_kernel_cache(std::move(cache)) {
    const auto get_host = [this]() {
        return h.get();
    };
    const auto wrap_snippets_emitter =
        [&](const auto& emitter,
            [[maybe_unused]] const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> {
#ifdef SNIPPETS_DEBUG_CAPS
        if (debug_config.enable_segfault_detector && is_segfault_detector_emitter(emitter.get())) {
            auto segfault_emitter = std::make_shared<intel_cpu::riscv64::jit_uni_segfault_detector_emitter>(
                h.get(),
                isa,
                emitter.get(),
                is_load_emitter(emitter.get()),
                is_store_emitter(emitter.get()),
                expr->get_node()->get_friendly_name());
            return std::make_shared<intel_cpu::riscv64::jit_debug_emitter>(
                emitter,
                segfault_emitter,
                intel_cpu::riscv64::jit_debug_emitter::EmissionLocation::preamble);
        }
#endif
        return emitter;
    };
    const auto emitter_factory = ov::intel_cpu::EmitterFactory{get_host, isa, wrap_snippets_emitter};

    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = emitter_factory.from_expr<jit_nop_emitter>();
    jitters[snippets::op::Result::get_type_info_static()] = emitter_factory.from_expr<jit_nop_emitter>();
    jitters[snippets::op::Scalar::get_type_info_static()] = emitter_factory.from_expr<jit_scalar_emitter>();

    // memory access
    jitters[snippets::op::Load::get_type_info_static()] = emitter_factory.from_expr<jit_load_memory_emitter>();
    jitters[snippets::op::LoadReorder::get_type_info_static()] = emitter_factory.from_expr<jit_load_memory_emitter>();
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] =
        emitter_factory.from_expr<jit_load_broadcast_emitter>();
    jitters[snippets::op::Store::get_type_info_static()] = emitter_factory.from_expr<jit_store_memory_emitter>();

    // loop control
    jitters[snippets::op::LoopBegin::get_type_info_static()] = emitter_factory.from_expr<jit_loop_begin_emitter>();
    jitters[snippets::op::LoopEnd::get_type_info_static()] = emitter_factory.from_expr<jit_loop_end_emitter>();

    // service kernel entry points
    jitters[snippets::op::KernelStatic::get_type_info_static()] =
        emitter_factory.from_expr<jit_kernel_static_emitter>();
    jitters[snippets::op::KernelDynamic::get_type_info_static()] =
        emitter_factory.from_expr<jit_kernel_dynamic_emitter>();

    // binary operations
    // jitters[op::v1::Add::get_type_info_static()] = emitter_factory.from_node<jit_add_emitter>();
    // jitters[op::v1::Divide::get_type_info_static()] = emitter_factory.from_node<jit_divide_emitter>();
    // jitters[op::v1::Maximum::get_type_info_static()] = emitter_factory.from_node<jit_maximum_emitter>();
    // jitters[op::v1::Minimum::get_type_info_static()] = emitter_factory.from_node<jit_minimum_emitter>();
    // jitters[op::v1::Mod::get_type_info_static()] = emitter_factory.from_node<jit_mod_emitter>();
    // jitters[op::v1::Multiply::get_type_info_static()] = emitter_factory.from_node<jit_multiply_emitter>();
    // jitters[snippets::op::PowerStatic::get_type_info_static()] =
    // emitter_factory.from_node<jit_power_static_emitter>(); jitters[op::v0::SquaredDifference::get_type_info_static()]
    // =
    //     emitter_factory.from_node<jit_squared_difference_emitter>();
    // jitters[op::v1::Subtract::get_type_info_static()] = emitter_factory.from_node<jit_subtract_emitter>();
    // jitters[op::v0::Xor::get_type_info_static()] = emitter_factory.from_node<jit_logical_xor_emitter>();

    // comparison operations
    // jitters[op::v1::Equal::get_type_info_static()] = emitter_factory.from_node<jit_equal_emitter>();
    // jitters[op::v1::Greater::get_type_info_static()] = emitter_factory.from_node<jit_greater_emitter>();
    // jitters[op::v1::GreaterEqual::get_type_info_static()] = emitter_factory.from_node<jit_greater_equal_emitter>();
    // jitters[op::v1::Less::get_type_info_static()] = emitter_factory.from_node<jit_less_emitter>();
    // jitters[op::v1::LessEqual::get_type_info_static()] = emitter_factory.from_node<jit_less_equal_emitter>();
    // jitters[op::v1::NotEqual::get_type_info_static()] = emitter_factory.from_node<jit_not_equal_emitter>();

    // logical operations
    // jitters[op::v1::LogicalAnd::get_type_info_static()] = emitter_factory.from_node<jit_logical_and_emitter>();
    // jitters[op::v1::LogicalOr::get_type_info_static()] = emitter_factory.from_node<jit_logical_or_emitter>();
    // jitters[op::v1::LogicalNot::get_type_info_static()] = emitter_factory.from_node<jit_logical_not_emitter>();
    // jitters[op::v1::LogicalXor::get_type_info_static()] = emitter_factory.from_node<jit_logical_xor_emitter>();

    // unary operations
    jitters[ov::op::v0::Abs::get_type_info_static()] = emitter_factory.from_node<jit_abs_emitter>();
    jitters[ov::op::v0::Clamp::get_type_info_static()] = emitter_factory.from_node<jit_clamp_emitter>();
    jitters[ov::op::v0::Elu::get_type_info_static()] = emitter_factory.from_node<jit_elu_emitter>();
    jitters[ov::op::v0::Erf::get_type_info_static()] = emitter_factory.from_node<jit_erf_emitter>();
    jitters[ov::op::v0::Exp::get_type_info_static()] = emitter_factory.from_node<jit_exp_emitter>();
    jitters[ov::op::v0::Floor::get_type_info_static()] = emitter_factory.from_node<jit_floor_emitter>();
    // jitters[ov::op::v1::FloorMod::get_type_info_static()] = emitter_factory.from_node<jit_floor_mod_emitter>();
    jitters[ov::op::v0::Gelu::get_type_info_static()] = emitter_factory.from_node<jit_gelu_erf_emitter>();
    // jitters[ov::op::v7::Gelu::get_type_info_static()] = emitter_factory.from_node<jit_gelu_erf_emitter>();
    jitters[ov::op::v5::HSigmoid::get_type_info_static()] = emitter_factory.from_node<jit_hsigmoid_emitter>();
    jitters[ov::op::v4::HSwish::get_type_info_static()] = emitter_factory.from_node<jit_hswish_emitter>();
    jitters[ov::op::v10::IsFinite::get_type_info_static()] = emitter_factory.from_node<jit_is_finite_emitter>();
    jitters[ov::op::v10::IsInf::get_type_info_static()] = emitter_factory.from_node<jit_is_inf_emitter>();
    jitters[ov::op::v10::IsNaN::get_type_info_static()] = emitter_factory.from_node<jit_is_nan_emitter>();
    jitters[ov::op::v4::Mish::get_type_info_static()] = emitter_factory.from_node<jit_mish_emitter>();
    jitters[ov::op::v0::Negative::get_type_info_static()] = emitter_factory.from_node<jit_negative_emitter>();
    // jitters[ov::op::v0::PRelu::get_type_info_static()] = emitter_factory.from_node<jit_prelu_emitter>();
    jitters[ov::op::v0::Relu::get_type_info_static()] = emitter_factory.from_node<jit_relu_emitter>();
    // jitters[ov::op::v5::Round::get_type_info_static()] =
    //     emitter_factory.from_node<jit_round_half_away_from_zero_emitter>();
    jitters[ov::op::v0::Sigmoid::get_type_info_static()] = emitter_factory.from_node<jit_sigmoid_emitter>();
    jitters[ov::op::v9::SoftSign::get_type_info_static()] = emitter_factory.from_node<jit_softsign_emitter>();
    jitters[ov::op::v0::Sqrt::get_type_info_static()] = emitter_factory.from_node<jit_sqrt_emitter>();
    jitters[ov::op::v0::Tanh::get_type_info_static()] = emitter_factory.from_node<jit_tanh_emitter>();
}

std::shared_ptr<ov::snippets::TargetMachine> CPUTargetMachine::clone() const {
    const auto cloned = std::make_shared<CPUTargetMachine>(isa, compiled_kernel_cache);
    cloned->configurator = std::make_shared<ov::snippets::RuntimeConfigurator>(*configurator);
#ifdef SNIPPETS_DEBUG_CAPS
    cloned->debug_config = debug_config;
#endif
    return cloned;
}

bool CPUTargetMachine::is_supported() const {
    return ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv);
}

snippets::CompiledSnippetPtr CPUTargetMachine::get_snippet() {
    OPENVINO_ASSERT(h->create_kernel(), "Failed to create jit_kernel in get_snippet()");

    const auto& result =
        std::make_shared<CompiledSnippetCPU>(std::unique_ptr<ov::intel_cpu::riscv64::jit_generator_t>(h.release()));
    // Note that we reset all the generated code, since it was copied into CompiledSnippetCPU
    h = std::make_unique<jit_snippet>();
    return result;
}

size_t CPUTargetMachine::get_lanes() const {
    switch (isa) {
    case ov::intel_cpu::riscv64::gv:
        // RISC-V Vector Extension lanes depend on VLEN, assume 128-bit VLEN with 32-bit elements
        return 4;  // 128-bit / 32-bit = 4 lanes for float32
    default:
        OPENVINO_THROW("unknown isa ", isa);
    }
}

std::vector<snippets::Reg> CPUTargetMachine::get_abi_arg_regs() const {
    // RISC-V ABI argument registers: a0-a7 (x10-x17)
    std::vector<snippets::Reg> res;
    for (size_t i = 0; i < 8; ++i) {
        res.emplace_back(snippets::RegType::gpr, 10 + i);  // a0-a7 are x10-x17
    }
    return res;
}

std::vector<snippets::Reg> CPUTargetMachine::get_gp_reg_pool() const {
    using Xbyak_riscv::Reg;
    const auto num_gp_regs = 32;
    std::vector<snippets::Reg> reg_pool;
    for (size_t i = 1; i < num_gp_regs; i++) {
        // Reserve: x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp), x5 (t0), x6 (t1), x8 (s0/fp)
        if (none_of(static_cast<int>(i),
                    Xbyak_riscv::ra.getIdx(),
                    Xbyak_riscv::sp.getIdx(),
                    Xbyak_riscv::gp.getIdx(),
                    Xbyak_riscv::tp.getIdx(),
                    Xbyak_riscv::t0.getIdx(),
                    Xbyak_riscv::t1.getIdx(),
                    Xbyak_riscv::s0.getIdx())) {
            reg_pool.emplace_back(snippets::RegType::gpr, i);
        }
    }
    return reg_pool;
}

std::vector<snippets::Reg> CPUTargetMachine::get_vec_reg_pool() const {
    const auto num_vec_regs = 32;  // RISC-V has 32 vector registers v0-v31
    std::vector<snippets::Reg> reg_pool;
    reg_pool.reserve(num_vec_regs);
    // v0 is typically reserved for masks, so exclude it
    for (int i = 1; i < num_vec_regs; i++) {
        reg_pool.emplace_back(snippets::RegType::vec, static_cast<size_t>(i));
    }
    return reg_pool;
}

ov::intel_cpu::riscv64::cpu_isa_t CPUTargetMachine::get_isa() const {
    return isa;
}

CPUGenerator::CPUGenerator(ov::intel_cpu::riscv64::cpu_isa_t isa_, ov::intel_cpu::MultiCacheWeakPtr cache)
    : Generator(std::make_shared<CPUTargetMachine>(isa_, std::move(cache))) {}
CPUGenerator::CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target) : Generator(target) {}

std::shared_ptr<ov::snippets::Generator> CPUGenerator::clone() const {
    const auto& cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    OPENVINO_ASSERT(cpu_target_machine,
                    "Failed to clone CPUGenerator: the instance contains incompatible TargetMachine type");
    return std::make_shared<CPUGenerator>(cpu_target_machine);
}

ov::snippets::RegType CPUGenerator::get_specific_op_out_reg_type(
    [[maybe_unused]] const ov::Output<ov::Node>& out) const {
    return ov::snippets::RegType::undefined;
}

bool CPUGenerator::uses_precompiled_kernel([[maybe_unused]] const std::shared_ptr<snippets::Emitter>& e) const {
    bool need = false;
#ifdef SNIPPETS_DEBUG_CAPS
    const auto cpu_target_machine = std::dynamic_pointer_cast<CPUTargetMachine>(target);
    need = need || (cpu_target_machine && cpu_target_machine->debug_config.enable_segfault_detector);
#endif
    return need;
}

}  // namespace intel_cpu::riscv64

}  // namespace ov
