// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief This transformation is part of Transpose/Gather sinking group of transformations.
 * This transformation moves Transpose through Concat layer. It changes Concat axis.
 * Currently GNA plugin has restrictions working with Concat layers.
 * It doesn't support all types of Concat layers. This transformation allows to remove Transpose layer
 * on the Concat input and replace it with Gather layer on the output ensuring supported Concat axis.
 * Substitute graph from
 *    Any#1 ... Any#K ... Any#N
 *      |         |         |
 *      |     Transpose     |
 *      |         |         |
 *             Concat
 *                |
 *             Any#M
 * to
 *    Any#1 ... Any#K ... Any#N
 *      |         |         |
 *    Reshape   Reshape   Reshape
 *      |         |         |
 *             Concat
 *                |
 *              Gather
 *                |
 *             Reshape
 *                |
 *             Any#M
 */
class TSConcatForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TSConcatForward", "0");
    TSConcatForward();
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
