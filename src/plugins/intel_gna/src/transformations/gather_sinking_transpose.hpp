// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

/**
 * @brief Merge Gather with Transpose through Reshape or sequence of Reshapes.
 *
 *      Any1[a,b,c]          Any1[a,b,c]
 *       |                     |
 *    Reshape[1,a*b*c]      Reshape[1,a*b*c]
 *       |                     |
 *     Gather[1,a*b*c]      Gather[1,a*b*c]
 *       |                     |
 *    Reshape1[...]            |
 *       |                     |
 *      ...                    |
 *       |                     |
 *    ReshapeN[a,b,c]          |
 *       |              =>     |
 *   Transpose[c,b,a]       Reshape[c,b,a]
 *       |                     |
 *      Any2[c,b,a]           Any2[c,b,a]
 *
 *  Gather restrictions:
 * - supported Scalar or 1D indexes
 *   i.e. [1, 64] or [64]
 */
class GatherSinkingTransposeForward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingTransposeForward", "0");
    GatherSinkingTransposeForward();
};

/**
 * @brief Merge Transpose with Gather through Reshape or sequence of Reshapes.
 *
 *      Any1[a,b,c]            Any1[a,b,c]
 *       |                      |
 *   Transpose[c,b,a]           |
 *       |                      |
 *    Reshape1[...]             |
 *       |                      |
 *      ...                     |
 *       |                      |
 *    Reshape[1, a*b*c]      Reshape[1, a*b*c]
 *       |               =>     |
 *     Gather[1, a*b*c]      Gather[1, a*b*c]
 *       |                      |
 *     Reshape[c,b,a]        Reshape[c,b,a]
 *       |                      |
 *      Any2[c,b,a]            Any2[c,b,a]
 *
 *  Gather restrictions:
 * - supported Scalar or 1D indexes
 *   i.e. [1, 64] or [64]
 */
class GatherSinkingTransposeBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("GatherSinkingTransposeBackward", "0");
    GatherSinkingTransposeBackward();
};

class GatherSinkingTranspose : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GatherSinkingTranspose", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
