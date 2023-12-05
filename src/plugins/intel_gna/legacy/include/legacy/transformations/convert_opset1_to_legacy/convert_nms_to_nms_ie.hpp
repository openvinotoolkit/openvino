// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <string>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertNMSToNMSIEMatcher;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     This transformation converts opset1::NonMaxSuppression to legacy NonMaxSuppressionIE
 *     NonMaxSuppressionIE takes max_output_boxes_per_class, iou_threshold and score_threshold
 *     inputs as 1D tensors when original operation requires scalars. And for this inputs
 *     we insert Unsqueeze operations.
 */

class ngraph::pass::ConvertNMSToNMSIEMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMSToNMSIEMatcher", "0");
    ConvertNMSToNMSIEMatcher();
};
