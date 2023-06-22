// Copyright (C) 2023 Intel Corporation2
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief
 * Fuses (merges) 2 consecutive Gather nodes into one. If resulting Gather node
 * does nothing with input (it does not actually permute items), just removes Gather nodes.
 *
 *   Any #1                  Any #1
 *     |                        |
 *   Gather #1              Gather #3
 *     |             =>         |
 *   Gather #2                Any #2
 *     |
 *    Any #2
 *
 *  or
 *
 *    Any #1                  Any #1
 *     |                        |
 *   Gather #1                Any #2
 *     |             =>
 *   Gather #2
 *     |
 *    Any #2
 *
 * This transformation is called called from GatherSinkingGeneral.
 */
class GatherSinkingFuse : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingFuse", "0");
    GatherSinkingFuse();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
