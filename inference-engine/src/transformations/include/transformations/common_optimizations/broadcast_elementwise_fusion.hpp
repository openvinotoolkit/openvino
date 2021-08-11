// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API BroadcastElementwiseFusion;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Removing Broadcast OP before ElementWise if output shape of Broadcast
 * are equal neighboring input shape of ElementWise.
 */

class ov::pass::BroadcastElementwiseFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    BroadcastElementwiseFusion();
};
