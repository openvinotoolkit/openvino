// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

/*
This transformation markup the 2nd/3rd inputs of Rope with FP32 to mantian accuracy.
+-------+    +-------+    +-------+
|intput1|    |input2 |    |input3 |
|(orig) |    |(fp32) |    |(fp32) |
+---|---+    +---|---+    +---|---+
    |            |            |
    |            |            |
 +--+------------|------------+--+
 |                               |
 |             ROPE              |
 +-------------------------------+
*/

class MarkUpRopeInputs : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkUpRopeInputs", "0");
    MarkUpRopeInputs();

private:
    std::unordered_set<std::shared_ptr<ov::Node>> visited;
};

}  // namespace intel_cpu
}  // namespace ov