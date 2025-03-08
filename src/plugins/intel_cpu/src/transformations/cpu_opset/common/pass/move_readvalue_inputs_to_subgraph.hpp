// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::intel_cpu {

/**
 * @brief Move ReadValue's inputs inside the new CPU ngraph node:ReadValueWithSubgraph op.
 *     intput1
 *        |
 *     Some nodes(They have only one common successor[ReadValue])     input1
 *        |                                                             |
 *     ReadValue                                            ------->  ReadValueWithSubgraph(Subgraph is inside)
 *        |     \                                                       |          \
 *     Assign   others                                                Assign       others
 */

class MoveReadValueInputsToSubgraph : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MoveReadValueInputsToSubgraph");
    MoveReadValueInputsToSubgraph();
};

}  // namespace ov::intel_cpu
