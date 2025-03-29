// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ExtractReshapesFromMHA
 * @brief This pass tries to extract unsupported reshape pairs around eltwise ops from MHA body
 * Subgraph inside the MHA body:
 *
 *               matmul
 *                 |
 *              Reshape1                      input1      input2
 *                 |   input1                       \     /
 *                 |  /                           ExtractedAdd
 *                Add1            =>                   |
 *                 |   input2                   ExtractedReshape
 *                 |  /                matmul      /
 *                Add2                       \    /
 *                 |                          Add
 *              Reshape2
 *
 * @ingroup snippets
 */
class ExtractReshapesFromMHA: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::ExtractReshapesFromMHA");
    ExtractReshapesFromMHA();
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
