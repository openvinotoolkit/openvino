// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ReshapeSequenceFusion;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ReshapeSequenceFusion fuses sequence of Reshape operation into single Reshape or eliminates full redundant
 * sequence
 */

class ngraph::pass::ReshapeSequenceFusion : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReshapeSequenceFusion", "0");
    ReshapeSequenceFusion(bool use_shape_for_elimination = true);
};
