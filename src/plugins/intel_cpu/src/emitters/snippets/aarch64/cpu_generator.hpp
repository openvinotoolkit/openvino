// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/aarch64/jit_generator.hpp"

#include "snippets/target_machine.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

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
    explicit CPUTargetMachine(dnnl::impl::cpu::aarch64::cpu_isa_t host_isa);
    std::shared_ptr<snippets::TargetMachine> clone() const override;
    bool is_supported() const override;
    snippets::CompiledSnippetPtr get_snippet() override;
    size_t get_lanes() const override;
    size_t get_reg_count() const override;
    dnnl::impl::cpu::aarch64::cpu_isa_t get_isa() const;

private:
    std::unique_ptr<dnnl::impl::cpu::aarch64::jit_generator> h;
    dnnl::impl::cpu::aarch64::cpu_isa_t isa;
};

class CPUGenerator : public snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::aarch64::cpu_isa_t isa);
    std::shared_ptr<Generator> clone() const override;

protected:
    bool uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& emitter) const override;
    ov::snippets::RegType get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const override;
};

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
