// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface ExtractLoopInvariants
 * @brief Extract the exprs that produce identical result in loop iteration to outside of loop
 * @ingroup snippets
 */
class ExtractLoopInvariants : public Pass {
public:
    OPENVINO_RTTI("ExtractLoopInvariants", "Pass")
    ExtractLoopInvariants() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
