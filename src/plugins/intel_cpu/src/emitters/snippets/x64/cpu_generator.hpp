// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"

#include "snippets/target_machine.hpp"
#include "snippets/generator.hpp"
#include "cache/multi_cache.h"
#include "snippets/runtime_configurator.hpp"

#include "emitters/snippets/jit_snippets_call_args.hpp"

#ifdef SNIPPETS_DEBUG_CAPS
#include "emitters/snippets/utils/debug_caps_config.hpp"
#endif

namespace ov {
namespace intel_cpu {

class CompiledSnippetCPU : public snippets::CompiledSnippet {
    const std::unique_ptr<const dnnl::impl::cpu::x64::jit_generator> h_compiled;
public:
    const uint8_t* get_code() const override;
    size_t get_code_size() const override;
    bool empty() const override;
    explicit CompiledSnippetCPU(std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h);
};

class CPUTargetMachine : public snippets::TargetMachine {
public:
    explicit CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                              ov::intel_cpu::MultiCacheWeakPtr);
    std::shared_ptr<snippets::TargetMachine> clone() const override;
    bool is_supported() const override;
    snippets::CompiledSnippetPtr get_snippet() override;
    size_t get_lanes() const override;
    size_t get_reg_count() const override;
    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const;
#ifdef SNIPPETS_DEBUG_CAPS
    SnippetsDebugCapsConfig debug_config;
#endif

private:
    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t isa;
    ov::intel_cpu::MultiCacheWeakPtr compiled_kernel_cache;
};

class CPUGenerator : public snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa, ov::intel_cpu::MultiCacheWeakPtr);
    CPUGenerator(const std::shared_ptr<CPUTargetMachine>& target);
    std::shared_ptr<Generator> clone() const override;

protected:
    ov::snippets::RegType get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const override;
    bool uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& emitter) const override;
};

}   // namespace intel_cpu
}   // namespace ov
