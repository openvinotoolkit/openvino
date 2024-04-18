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
 * @interface ValidateExpandedLoops
 * @brief The pass validates loops after decomposition into specific iterations
 * @ingroup snippets
 */
class ValidateExpandedLoops : public Pass {
public:
    OPENVINO_RTTI("ValidateExpandedLoops", "Pass")
    ValidateExpandedLoops() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
