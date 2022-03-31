// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <cpu/x64/jit_generator.hpp>

#include "snippets/generator.hpp"

namespace ov {
namespace intel_cpu {

class CPUTargetMachine : public ngraph::snippets::TargetMachine {
public:
    CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa);

    bool is_supported() const override;
    ngraph::snippets::code get_snippet() const override;
    size_t get_lanes() const override;

private:
    std::unique_ptr<dnnl::impl::cpu::x64::jit_generator> h;
    dnnl::impl::cpu::x64::cpu_isa_t isa;
};

class CPUGenerator : public ngraph::snippets::Generator {
public:
    CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa);
    std::vector<size_t> get_gprs_for_data_pointers() const override;
};

}   // namespace intel_cpu
}   // namespace ov
