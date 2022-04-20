// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API BroadcastConstRangeReplacement;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief BroadcastConstRangeReplacement replaces Constant filled with range values starting from 0 and replaces it with
 * Range op
 */

class ngraph::pass::BroadcastConstRangeReplacement : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("BroadcastConstRangeReplacement", "0");
    BroadcastConstRangeReplacement();
};
