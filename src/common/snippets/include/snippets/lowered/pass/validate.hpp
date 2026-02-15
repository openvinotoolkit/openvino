// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <unordered_map>

#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"
#include "pass.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

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

}  // namespace ov::snippets::lowered::pass
