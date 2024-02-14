// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DecomposeIntegerFloorDivide;
/**
 * @ingroup ie_transformation_common_api
 * @brief DecomposeIntegerFloorDivide transformation decomposes floor_div ops (Divide + Floor) for integer data types.
 * @details When floor_div is performed for integers with different signs reminder should be taken into account.
 * floor_div = x / y; if x > 0 and y > 0
 * floor_div = x / y - 1; if (x < 0 xor y < 0) and (x mod y != 0)
 * Div or / is a truncated  divide. To perform expression above this pass inserts neccesarry Mod, LogicalXor, Select,
 * etc. ops.
 */
class DecomposeIntegerFloorDivide : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DecomposeIntegerFloorDivide", "0");
    DecomposeIntegerFloorDivide();
};

}  // namespace pass
}  // namespace ov
