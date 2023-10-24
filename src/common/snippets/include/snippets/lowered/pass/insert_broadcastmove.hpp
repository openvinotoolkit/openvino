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
class InsertBroadcastMove : public Pass {
public:
    OPENVINO_RTTI("InsertBroadcastMove", "Pass")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
