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
/**
 * @class RuntimeOptimizer
 * @brief Base class for runtime optimizers that operate on LinearIR and RuntimeConfigurator during
 * RuntimeConfigurator::update stage.
 */
class RuntimeOptimizer : public ConstPass {
public:
    RuntimeOptimizer() = default;
    RuntimeOptimizer(RuntimeConfigurator* configurator) : m_configurator(configurator) {}
protected:
    RuntimeConfigurator* m_configurator = nullptr;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
