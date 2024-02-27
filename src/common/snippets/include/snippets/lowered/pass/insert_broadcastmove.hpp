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
 * @interface InsertMovebroadcast
 * @brief Injects explicit Movebroadcast operations when the most varying dim is broadcasted
 * @ingroup snippets
 */
class InsertBroadcastMove : public RangedPass {
public:
    OPENVINO_RTTI("InsertBroadcastMove", "RangedPass")
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
