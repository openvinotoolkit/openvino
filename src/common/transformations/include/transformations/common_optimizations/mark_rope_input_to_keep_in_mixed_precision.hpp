// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief This transformation markups the 2nd/3rd inputs of Rope with FP32 to mantian accuracy.
 * +-------+    +-------+    +-------+
 * |intput1|    |input2 |    |input3 |
 * |(orig) |    |(fp32) |    |(fp32) |
 * +---|---+    +---|---+    +---|---+
 *     |            |            |
 *     |            |            |
 *  +--+------------|------------+--+
 *  |                               |
 *  |             ROPE              |
 *  +-------------------------------+
 */

class TRANSFORMATIONS_API MarkRopeInputsToKeepInMixedPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkRopeInputsToKeepInMixedPrecision", "0");
    MarkRopeInputsToKeepInMixedPrecision();

private:
    std::unordered_set<ov::Node*> visited;
};

}  // namespace pass
}  // namespace ov