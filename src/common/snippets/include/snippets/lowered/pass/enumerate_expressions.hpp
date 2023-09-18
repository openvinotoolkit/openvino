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
 * @interface EnumerateExpressions
 * @brief The pass enumerates expression by execution order
 * @ingroup snippets
 */
class EnumerateExpressions : public Pass {
public:
    OPENVINO_RTTI("EnumerateExpressions", "Pass")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
