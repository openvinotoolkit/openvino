// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertNMSToNMSIE);

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 *     This transformation converts opset1::NonMaxSuppression to legacy NonMaxSuppressionIE
 *     NonMaxSuppressionIE takes max_output_boxes_per_class, iou_threshold and score_threshold
 *     inputs as 1D tensors when original operation requires scalars. And for this inputs
 *     we insert Unsqueeze operations.
 */

class ngraph::pass::ConvertNMSToNMSIE : public ngraph::pass::GraphRewrite {
public:
    ConvertNMSToNMSIE() : GraphRewrite() {
        convert_nms_to_nms_ie();
    }

private:
    void convert_nms_to_nms_ie();
};
