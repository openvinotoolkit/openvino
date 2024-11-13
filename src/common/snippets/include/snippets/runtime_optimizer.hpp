// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/runtime_configurator.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
class RuntimeOptimizer : public Pass {
public:
    RuntimeOptimizer() = default;
    RuntimeOptimizer(RuntimeConfigurator* configurator) : m_configurator(configurator) {}

    virtual bool run(const snippets::lowered::LinearIR& linear_ir) = 0;
    bool run(snippets::lowered::LinearIR& linear_ir) override final { // NOLINT
        return run(const_cast<const snippets::lowered::LinearIR&>(linear_ir));
    }

protected:
    RuntimeConfigurator* m_configurator = nullptr;
};

class RuntimeOptimizersPipeline : public PassPipeline {
public:
    void run(const lowered::LinearIR& linear_ir) const {
        PassPipeline::run(const_cast<snippets::lowered::LinearIR&>(linear_ir));
    }
};
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
