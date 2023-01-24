// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateUnsqueezeGather;
class TRANSFORMATIONS_API EliminateGatherUnsqueeze;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Unsqueeze + Gather pair, if Gather gathers data by dimension
 * that was previously added by Unsqueeze
 */

class ov::pass::EliminateUnsqueezeGather : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateUnsqueezeGather", "0");
    EliminateUnsqueezeGather();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Remove Gather -> Unsqueeze pair, if Gather takes a scalar and
 * Unsqueeze makes it a 1D tensor
 */

class ov::pass::EliminateGatherUnsqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateGatherUnsqueeze", "0");
    EliminateGatherUnsqueeze();
};

namespace ngraph {
namespace pass {
using ov::pass::EliminateGatherUnsqueeze;
using ov::pass::EliminateUnsqueezeGather;
}  // namespace pass
}  // namespace ngraph
