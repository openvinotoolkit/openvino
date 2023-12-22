// Copyright (C) 2018-2023 Intel Corporation
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
class LoadMoveBroadcastToBroadcastLoad: public Pass {
public:
    LoadMoveBroadcastToBroadcastLoad() = default;
    OPENVINO_RTTI("LoadMoveBroadcastToBroadcastLoad", "Pass")
    bool run(LinearIR& linear_ir) override;
};

}  // namespace pass
}  // namespace lowered
}  // namespace snippets
}  // namespace ov
