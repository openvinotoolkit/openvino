// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

#include "ngraph/pattern/matcher.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API BroadcastElementwiseFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Removing Broadcast OP before ElementWise if output shape of Broadcast
 * are equal neighboring input shape of ElementWise.
 */

class ngraph::pass::BroadcastElementwiseFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastElementwiseFusion", "0");
    BroadcastElementwiseFusion();
};
