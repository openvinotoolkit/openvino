// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu/x64/jit_generator.hpp"

#include "snippets/target_machine.hpp"
#include "snippets/generator.hpp"
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

class CPURuntimeConfig : public ov::snippets::RuntimeConfig {
public:
    CPURuntimeConfig() = default;

    size_t tensor_rank = 0;
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
    std::vector<ov::snippets::VectorDims> io_data_offsets = {};
    ov::snippets::VectorDims parallel_domain = {};
};

class CPURuntimeConfigurator : public ov::snippets::RuntimeConfigurator {
public:
    CPURuntimeConfigurator();

protected:
    bool is_update_needed(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) override;
    void update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) override;

    void init_data_info(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir);
    void update_data_offsets(const std::shared_ptr<CPURuntimeConfig>& cpu_config) const;
    void update_loop_args(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                          const std::shared_ptr<CPURuntimeConfig>& cpu_config) const;
    void update_parallel_domain(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                const std::shared_ptr<CPURuntimeConfig>& cpu_config) const;
    void update_latest_shapes();

    const size_t rank6D = 6;

    size_t m_io_num = 0;
    size_t m_in_num = 0;
    std::vector<ov::snippets::VectorDimsPtr> m_io_shapes = {};
    std::vector<std::vector<size_t>> m_io_layouts = {};
    std::vector<size_t> m_io_data_sizes = {};

    std::vector<ov::snippets::VectorDims> m_latest_input_shapes = {};
};

class CPUTargetMachine : public snippets::TargetMachine {
public:
    explicit CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa);

    bool is_supported() const override;
    snippets::CompiledSnippetPtr get_snippet() override;
    size_t get_lanes() const override;
    dnnl::impl::cpu::x64::cpu_isa_t get_isa() const;
#ifdef SNIPPETS_DEBUG_CAPS
    SnippetsDebugCapsConfig debug_config;
#endif

private:
    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t isa;
};

class CPUGenerator : public snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa);
    std::shared_ptr<Generator> clone() const override;

protected:
    ov::snippets::RegType get_specific_op_out_reg_type(const ov::Output<ov::Node>& out) const override;
    bool uses_precompiled_kernel(const std::shared_ptr<snippets::Emitter>& emitter) const override;
};

}   // namespace intel_cpu
}   // namespace ov
