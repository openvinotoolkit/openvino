// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

#include "snippets/lowered/loop_manager.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface ValidateShapes
 * @brief The pass checks that there are no dynamic shapes in the IR
 * @ingroup snippets
 */
class ValidateShapes : public Pass {
public:
    OPENVINO_RTTI("ValidateShapes", "Pass")
    ValidateShapes() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
