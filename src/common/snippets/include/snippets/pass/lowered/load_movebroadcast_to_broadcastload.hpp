// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "linear_IR_transformation.hpp"

namespace ngraph {
namespace snippets {
namespace pass {
namespace lowered {

/**
 * @interface LoadMoveBroadcastToBroadcastLoad
 * @brief Fuses consecutive Load and MoveBroadcast into a single load insctruction.
 * @ingroup snippets
 */
class LoadMoveBroadcastToBroadcastLoad: public LinearIRTransformation {
public:
    LoadMoveBroadcastToBroadcastLoad() = default;
    OPENVINO_RTTI("LoadMoveBroadcastToBroadcastLoad", "LinearIRTransformation")
    bool run(LoweredExprIR& linear_ir) override;
};

}  // namespace lowered
}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
