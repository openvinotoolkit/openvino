// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace snippets {
// TODO: inherit from lowered pass?
class RuntimeOptimizer {
public:
    RuntimeOptimizer() = default;
    RuntimeOptimizer(RuntimeConfigurator* configurator) : m_configurator(configurator) {}

    virtual bool optimize(const ov::snippets::lowered::LinearIRCPtr& linear_ir) = 0;

protected:
    RuntimeConfigurator* m_configurator = nullptr;
};
} // namespace snippets
} // namespace ov
