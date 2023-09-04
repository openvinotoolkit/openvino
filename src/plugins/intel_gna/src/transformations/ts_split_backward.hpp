// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief This transformation is a part of Transpose/Gather sinking group of transformations.
 * This transformation moves Transpose through Split layer. It changes Split axis.
 * Currently GNA plugin has restrictions working with Split layers.
 * It doesn't support any Split layers. This transformation allows to remove Transpose layer
 * on the Split output and replace it with a Gather layer on the input with ensuring supported Split axis.
 * Substitute from
 *          Any#1
 *           |
 *         Split
 *    |      |        |
 *    |   Transpose   |
 *    |      |        |
 * Any#2 .. Any#K .. Any#N
 * to
 *          Any#1
 *           |
 *         Reshape
 *           |
 *         Gather
 *           |
 *         Split
 *    |      |        |
 * Reshape Reshape   Reshape
 *    |      |        |
 * Any#2 .. Any#K .. Any#N
 */
class TSSplitBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TSSplitBackward", "0");
    TSSplitBackward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
