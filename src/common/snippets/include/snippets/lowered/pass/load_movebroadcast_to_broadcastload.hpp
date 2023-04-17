// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformation.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface LoadMoveBroadcastToBroadcastLoad
 * @brief Fuses consecutive Load and MoveBroadcast into a single load insctruction.
 * @ingroup snippets
 */
class LoadMoveBroadcastToBroadcastLoad: public Transformation {
public:
    LoadMoveBroadcastToBroadcastLoad() = default;
    OPENVINO_RTTI("LoadMoveBroadcastToBroadcastLoad", "Transformation")
    bool run(LinearIR& linear_ir) override;
};

}  // namespace pass
}  // namespace lowered
}  // namespace snippets
}  // namespace ngraph
