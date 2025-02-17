// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "cpu/aarch64/jit_generator.hpp"
#include "snippets/generator.hpp"
#include "snippets/target_machine.hpp"

namespace ov::intel_cpu::aarch64 {

class CompiledSnippetCPU : public snippets::CompiledSnippet {
public:
    explicit CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h);
    const uint8_t* get_code() const override;
    size_t get_code_size() const override;
    bool empty() const override;

private:
    const std::unique_ptr<const dnnl::impl::cpu::aarch64::jit_generator> h_compiled;
};

class CPUTargetMachine : public snippets::TargetMachine {
public:
    explicit CPUTargetMachine(dnnl::impl::cpu::aarch64::cpu_isa_t host_isa, ov::intel_cpu::MultiCacheWeakPtr);
    std::shared_ptr<snippets::TargetMachine> clone() const override;
    bool is_supported() const override;
    snippets::CompiledSnippetPtr get_snippet() override;
    size_t get_lanes() const override;

    std::vector<snippets::Reg> get_abi_arg_regs() const override;
    std::vector<snippets::Reg> get_gp_reg_pool() const override;
    std::vector<snippets::Reg> get_vec_reg_pool() const override;

    dnnl::impl::cpu::aarch64::cpu_isa_t get_isa() const;

private:
    std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h;
    dnnl::impl::cpu::aarch64::cpu_isa_t isa;
    ov::intel_cpu::MultiCacheWeakPtr compiled_kernel_cache;
};

class CPUGenerator : public snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::aarch64::cpu_isa_t isa, ov::intel_cpu::MultiCacheWeakPtr);
    CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target);
    std::shared_ptr<Generator> clone() const override;

protected:
    bool uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& emitter) const override;
    ov::snippets::RegType get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const override;
};

}  // namespace ov::intel_cpu::aarch64
