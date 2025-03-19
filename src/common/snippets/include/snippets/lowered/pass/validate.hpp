// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface Validate
 * @brief The pass validates Linear IR state after all passes (on the final transformation stage):
 * @ingroup snippets
 */
class Validate : public RangedPass {
public:
    OPENVINO_RTTI("Validate", "", RangedPass)
    Validate();
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;

private:
    using Validator = std::function<void(const ExpressionPtr& expr, const LinearIR& linear_ir)>;
    std::unordered_map<ov::DiscreteTypeInfo, Validator> m_validation_map;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
