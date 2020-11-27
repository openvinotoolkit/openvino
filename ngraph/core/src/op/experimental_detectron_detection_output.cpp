//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/experimental_detectron_detection_output.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

using ExperimentalDetection = op::v6::ExperimentalDetectronDetectionOutput;

NGRAPH_RTTI_DEFINITION(ExperimentalDetection, "ExperimentalDetectronDetectionOutput", 6);

ExperimentalDetection::ExperimentalDetectronDetectionOutput(const Output<Node>& input_rois,
                                                            const Output<Node>& input_deltas,
                                                            const Output<Node>& input_scores,
                                                            const Output<Node>& input_im_info,
                                                            const Attributes& attrs)
    : Op({input_rois, input_deltas, input_scores, input_im_info})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

bool op::ExperimentalDetectronDetectionOutput::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("max_delta_log_wh", m_attrs.max_delta_log_wh);
    visitor.on_attribute("num_classes", m_attrs.num_classes);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("max_detections_per_image", m_attrs.max_detections_per_image);
    visitor.on_attribute("class_agnostic_box_regression", m_attrs.class_agnostic_box_regression);
    visitor.on_attribute("deltas_weights", m_attrs.deltas_weights);
    return true;
}

void op::ExperimentalDetectronDetectionOutput::validate_and_infer_types()
{
    size_t rois_num = static_cast<size_t>(m_attrs.max_detections_per_image);
    auto input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(
        this,
        inputs().size() != 4,
        "Number of inputs of ExperimentalDetectronDetectionOutput must be equal to 4, "
        "but number of inputs is ",
        inputs().size());

    set_output_type(0, input_et, Shape{rois_num, 4});
    set_output_type(1, element::i32, Shape{rois_num});
    set_output_type(2, input_et, Shape{rois_num});
}

shared_ptr<Node> ExperimentalDetection::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);

    return make_shared<op::v6::ExperimentalDetectronDetectionOutput>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_attrs);
}
