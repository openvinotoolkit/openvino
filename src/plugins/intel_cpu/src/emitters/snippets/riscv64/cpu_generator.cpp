// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_generator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <set>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/riscv64/jit_eltwise_emitters.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "jit_kernel_emitter.hpp"
#include "jit_loop_emitters.hpp"
#include "jit_memory_emitters.hpp"
#include "jit_snippets_emitters.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/kernel.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"
#include "snippets/target_machine.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov {

#define CREATE_SNIPPETS_EMITTER(e_type, ...)                                                      \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return std::make_shared<e_type>(h.get(), isa, expr, ##__VA_ARGS__);                      \
     },                                                                                           \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         return e_type::get_supported_precisions(n);                                              \
     }}

#define CREATE_CPU_EMITTER(e_type)                                                                \
    {[this](const snippets::lowered::ExpressionPtr& expr) -> std::shared_ptr<snippets::Emitter> { \
         return std::make_shared<e_type>(h.get(), isa, expr->get_node());                         \
     },                                                                                           \
     [](const std::shared_ptr<ov::Node>& n) -> std::set<std::vector<element::Type>> {             \
         return e_type::get_supported_precisions(n);                                              \
     }}

class jit_snippet : public ov::intel_cpu::riscv64::jit_generator_t {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() override = default;

    jit_snippet() = default;

    void generate() override {}
};

namespace intel_cpu::riscv64 {

CompiledSnippetCPU::CompiledSnippetCPU(std::unique_ptr<ov::intel_cpu::riscv64::jit_generator_t> h)
    : h_compiled(std::move(h)) {
    OPENVINO_ASSERT(h_compiled && h_compiled->jit_ker(), "Got invalid jit generator or kernel was not compiled");
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

CPUTargetMachine::CPUTargetMachine(ov::intel_cpu::riscv64::cpu_isa_t host_isa, ov::intel_cpu::MultiCacheWeakPtr cache)
    : TargetMachine(std::make_shared<CPURuntimeConfigurator>(cache)),
      h(new jit_snippet()),
      isa(host_isa),
      compiled_kernel_cache(std::move(cache)) {
    // data movement
    jitters[op::v0::Parameter::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[op::v0::Result::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_nop_emitter);
    jitters[snippets::op::Scalar::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_scalar_emitter);

    // memory access
    jitters[snippets::op::Load::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_memory_emitter);
    jitters[snippets::op::LoadReorder::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_memory_emitter);
    jitters[snippets::op::BroadcastLoad::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_load_broadcast_emitter);
    jitters[snippets::op::Store::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_store_memory_emitter);

    // loop control
    jitters[snippets::op::LoopBegin::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_loop_begin_emitter);
    jitters[snippets::op::LoopEnd::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_loop_end_emitter);

    // service kernel entry points
    jitters[snippets::op::KernelStatic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_static_emitter);
    jitters[snippets::op::KernelDynamic::get_type_info_static()] = CREATE_SNIPPETS_EMITTER(jit_kernel_dynamic_emitter);

    // binary operations
    jitters[op::v1::Add::get_type_info_static()] = CREATE_CPU_EMITTER(ov::intel_cpu::riscv64::jit_add_emitter);
}

std::shared_ptr<snippets::TargetMachine> CPUTargetMachine::clone() const {
    return std::make_shared<CPUTargetMachine>(isa, compiled_kernel_cache);
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
        // Reserve: x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp), x8 (s0/fp)
        if (none_of(i,
                    Xbyak_riscv::ra.getIdx(),
                    Xbyak_riscv::sp.getIdx(),
                    Xbyak_riscv::gp.getIdx(),
                    Xbyak_riscv::tp.getIdx(),
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
    for (int i = 0; i < num_vec_regs; i++) {
        // v0 is typically reserved for masks, so exclude it
        if (i != 0) {
            reg_pool.emplace_back(snippets::RegType::vec, static_cast<size_t>(i));
        }
    }
    return reg_pool;
}

ov::intel_cpu::riscv64::cpu_isa_t CPUTargetMachine::get_isa() const {
    return isa;
}

CPUGenerator::CPUGenerator(ov::intel_cpu::riscv64::cpu_isa_t isa_, ov::intel_cpu::MultiCacheWeakPtr cache)
    : Generator(std::make_shared<CPUTargetMachine>(isa_, std::move(cache))) {}
CPUGenerator::CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target) : Generator(target) {}

std::shared_ptr<snippets::Generator> CPUGenerator::clone() const {
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
    // RISC-V platform doesn't currently use precompiled kernels
    return false;
}

}  // namespace intel_cpu::riscv64

}  // namespace ov
