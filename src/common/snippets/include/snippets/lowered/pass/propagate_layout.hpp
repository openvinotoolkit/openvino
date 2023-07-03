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
 * @interface PropagateLayout
 * @brief Propagate layout from Parameter child to parameter and from Result Parent to Result. This is needed to calculate
 * proper data pointer offsets in the Kernel;
 * @ingroup snippets
 */
class PropagateLayout : public Pass {
public:
    OPENVINO_RTTI("PropagateLayout", "Pass")
    bool run(LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
