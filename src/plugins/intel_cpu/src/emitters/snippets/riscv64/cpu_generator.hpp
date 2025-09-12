// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <vector>

#include "cache/multi_cache.h"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "snippets/emitter.hpp"
#include "snippets/generator.hpp"
#include "snippets/target_machine.hpp"

namespace ov::intel_cpu::riscv64 {

class CompiledSnippetCPU : public snippets::CompiledSnippet {
public:
    explicit CompiledSnippetCPU(std::unique_ptr<ov::intel_cpu::riscv64::jit_generator_t> h);
    [[nodiscard]] const uint8_t* get_code() const override;
    [[nodiscard]] size_t get_code_size() const override;
    [[nodiscard]] bool empty() const override;

private:
    const std::unique_ptr<const ov::intel_cpu::riscv64::jit_generator_t> h_compiled;
};

class CPUTargetMachine : public snippets::TargetMachine {
public:
    explicit CPUTargetMachine(ov::intel_cpu::riscv64::cpu_isa_t host_isa, ov::intel_cpu::MultiCacheWeakPtr cache);
    [[nodiscard]] std::shared_ptr<snippets::TargetMachine> clone() const override;
    [[nodiscard]] bool is_supported() const override;
    snippets::CompiledSnippetPtr get_snippet() override;
    [[nodiscard]] size_t get_lanes() const override;

    [[nodiscard]] std::vector<snippets::Reg> get_abi_arg_regs() const override;
    [[nodiscard]] std::vector<snippets::Reg> get_gp_reg_pool() const override;
    [[nodiscard]] std::vector<snippets::Reg> get_vec_reg_pool() const override;

    [[nodiscard]] ov::intel_cpu::riscv64::cpu_isa_t get_isa() const;

private:
    std::unique_ptr<ov::intel_cpu::riscv64::jit_generator_t> h;
    ov::intel_cpu::riscv64::cpu_isa_t isa;
    ov::intel_cpu::MultiCacheWeakPtr compiled_kernel_cache;
};

class CPUGenerator : public snippets::Generator {
public:
    CPUGenerator(ov::intel_cpu::riscv64::cpu_isa_t isa, ov::intel_cpu::MultiCacheWeakPtr cache);
    CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target);
    std::shared_ptr<Generator> clone() const override;

protected:
    bool uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& emitter) const override;
    ov::snippets::RegType get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const override;
};

}  // namespace ov::intel_cpu::riscv64