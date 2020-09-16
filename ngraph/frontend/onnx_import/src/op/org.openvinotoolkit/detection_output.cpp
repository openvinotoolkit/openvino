//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/detection_output.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/org.openvinotoolkit/detection_output.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector detection_output(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();

                    auto box_logits = inputs[0];
                    auto class_preds = inputs[1];
                    auto proposals = inputs[2];

                    ngraph::op::DetectionOutputAttrs attrs;
                    attrs.num_classes = node.get_attribute_value<int64_t>("num_classes");
                    attrs.background_label_id =
                        node.get_attribute_value<int64_t>("background_label_id", 0);
                    attrs.top_k = node.get_attribute_value<int64_t>("top_k", -1);
                    attrs.variance_encoded_in_target =
                        node.get_attribute_value<int64_t>("variance_encoded_in_target", 0);
                    // spec says keep_top_k is an array of ints, but some models use a single int
                    // also mkldnn expects single integer
                    attrs.keep_top_k = {
                        static_cast<int>(node.get_attribute_value<int64_t>("keep_top_k", 1))};

                    auto code_type = node.get_attribute_value<std::string>(
                        "code_type", std::string{"caffe.PriorBoxParameter.CORNER"});
                    // possible values are "caffe.PriorBoxParameter.CENTER_SIZE",
                    // "caffe.PriorBoxParameter.CORNER"
                    // but we can just have "CENTER_SIZE" or "CORNER" strings here, so we need to
                    // handle that case
                    if (code_type.find("caffe.PriorBoxParameter.") == std::string::npos)
                    {
                        code_type = "caffe.PriorBoxParameter." + code_type;
                    }
                    attrs.code_type = code_type;
                    attrs.share_location = node.get_attribute_value<int64_t>("share_location", 1);
                    attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold");
                    attrs.confidence_threshold =
                        node.get_attribute_value<float>("confidence_threshold", 0);
                    attrs.clip_after_nms = node.get_attribute_value<int64_t>("clip_after_nms", 0);
                    attrs.clip_before_nms = node.get_attribute_value<int64_t>("clip_before_nms", 0);
                    attrs.decrease_label_id =
                        node.get_attribute_value<int64_t>("decrease_label_id", 0);
                    // TODO: per spec, normalized by default should be 0, but in MO it's 1.
                    attrs.normalized = node.get_attribute_value<int64_t>("normalized", 1);
                    attrs.input_width = node.get_attribute_value<int64_t>("input_width", 1);
                    attrs.input_height = node.get_attribute_value<int64_t>("input_height", 1);
                    attrs.objectness_score = node.get_attribute_value<float>("objectness_score", 0);

                    if (inputs.size() == 3)
                    {
                        return {std::make_shared<default_opset::DetectionOutput>(
                            box_logits, class_preds, proposals, attrs)};
                    }
                    else if (inputs.size() == 5)
                    {
                        auto aux_class_preds = inputs[3];
                        auto aux_box_preds = inputs[4];
                        return {std::make_shared<default_opset::DetectionOutput>(box_logits,
                                                                                 class_preds,
                                                                                 proposals,
                                                                                 aux_class_preds,
                                                                                 aux_box_preds,
                                                                                 attrs)};
                    }
                    else
                    {
                        NGRAPH_CHECK(false, "Invalid number of inputs");
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
