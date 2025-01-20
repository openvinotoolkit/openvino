// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pass.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface LoadMoveBroadcastToBroadcastLoad
 * @brief Fuses consecutive Load and MoveBroadcast into a single load insctruction.
 * @ingroup snippets
 */
class LoadMoveBroadcastToBroadcastLoad: public RangedPass {
public:
    LoadMoveBroadcastToBroadcastLoad() = default;
    OPENVINO_RTTI("LoadMoveBroadcastToBroadcastLoad", "", RangedPass);
    bool run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) override;
};

}  // namespace pass
}  // namespace lowered
}  // namespace snippets
}  // namespace ov
