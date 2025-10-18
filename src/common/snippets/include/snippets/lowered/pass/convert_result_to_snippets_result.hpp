// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/rtti.hpp"
#include "pass.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::snippets::lowered::pass {

/**
 * @interface ConvertResultToSnippetsResult
 * @brief Convert result to Snippets result.
 * @ingroup snippets
 */
class ConvertResultToSnippetsResult : public RangedPass {
public:
    ConvertResultToSnippetsResult() = default;
    OPENVINO_RTTI("ConvertResultToSnippetsResult", "", RangedPass);
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

}  // namespace ov::snippets::lowered::pass
