// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/runtime_configurator.hpp"

#include "snippets/lowered/port_descriptor.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

#include "memory_desc/cpu_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {

class CPURuntimeConfigurator;
class BrgemmExternalRepackingAdjuster {
public:
    BrgemmExternalRepackingAdjuster() = default;
    BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir, CPURuntimeConfigurator* configurator);

    void optimize(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                  const std::vector<ov::snippets::VectorDims>& shapes,
                  const std::vector<std::vector<size_t>>& layouts);

private:
    CPURuntimeConfigurator* m_configurator = nullptr;
    std::set<size_t> m_param_idces_with_external_repacking;
};

}   // namespace intel_cpu
}   // namespace ov
