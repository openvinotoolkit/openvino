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
 * @interface MoveResultOutOfLoop
 * @brief After passes with Loop work Result expressions might be inside Loop.
 *        It means that Result can be before his Parent and LoopEnd, this situation breaks control dependency and
 *        create cycle dependency in AssignRegister algorithm.
 *        The pass extracts Result expressions from Loop and insert after.
 * @ingroup snippets
 */
class MoveResultOutOfLoop : public Pass {
public:
    OPENVINO_RTTI("MoveResultOutOfLoop", "", Pass);
    MoveResultOutOfLoop() = default;
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
