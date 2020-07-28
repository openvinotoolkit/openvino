// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertNMSToNMSIEMatcher;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     This transformation converts opset1::NonMaxSuppression to legacy NonMaxSuppressionIE
 *     NonMaxSuppressionIE takes max_output_boxes_per_class, iou_threshold and score_threshold
 *     inputs as 1D tensors when original operation requires scalars. And for this inputs
 *     we insert Unsqueeze operations.
 */

class ngraph::pass::ConvertNMSToNMSIEMatcher : public ngraph::pass::MatcherPass {
public:
    ConvertNMSToNMSIEMatcher();
};
