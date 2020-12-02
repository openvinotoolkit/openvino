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

#include "ngraph/op/detection_output.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DetectionOutput::type_info;

op::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                     const Output<Node>& class_preds,
                                     const Output<Node>& proposals,
                                     const Output<Node>& aux_class_preds,
                                     const Output<Node>& aux_box_preds,
                                     const DetectionOutputAttrs& attrs)
    : Op({box_logits, class_preds, proposals, aux_class_preds, aux_box_preds})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

op::DetectionOutput::DetectionOutput(const Output<Node>& box_logits,
                                     const Output<Node>& class_preds,
                                     const Output<Node>& proposals,
                                     const DetectionOutputAttrs& attrs)
    : Op({box_logits, class_preds, proposals})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::DetectionOutput::validate_and_infer_types()
{
    if (get_input_partial_shape(0).is_static())
    {
        auto box_logits_shape = get_input_partial_shape(0).to_shape();
        set_output_type(
            0, element::Type_t::f32, Shape{1, 1, m_attrs.keep_top_k[0] * box_logits_shape[0], 7});
    }
    else
    {
        set_output_type(0, element::Type_t::f32, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::DetectionOutput::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);

    auto num_args = new_args.size();

    NODE_VALIDATION_CHECK(
        this, num_args == 3 || num_args == 5, "DetectionOutput accepts 3 or 5 inputs.");

    if (num_args == 3)
    {
        return make_shared<DetectionOutput>(
            new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    }
    else
    {
        return make_shared<DetectionOutput>(new_args.at(0),
                                            new_args.at(1),
                                            new_args.at(2),
                                            new_args.at(3),
                                            new_args.at(4),
                                            m_attrs);
    }
}

bool op::DetectionOutput::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("num_classes", m_attrs.num_classes);
    visitor.on_attribute("background_label_id", m_attrs.background_label_id);
    visitor.on_attribute("top_k", m_attrs.top_k);
    visitor.on_attribute("variance_encoded_in_target", m_attrs.variance_encoded_in_target);
    visitor.on_attribute("keep_top_k", m_attrs.keep_top_k);
    visitor.on_attribute("code_type", m_attrs.code_type);
    visitor.on_attribute("share_location", m_attrs.share_location);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("confidence_threshold", m_attrs.confidence_threshold);
    visitor.on_attribute("clip_after_nms", m_attrs.clip_after_nms);
    visitor.on_attribute("clip_before_nms", m_attrs.clip_before_nms);
    visitor.on_attribute("decrease_label_id", m_attrs.decrease_label_id);
    visitor.on_attribute("normalized", m_attrs.normalized);
    visitor.on_attribute("input_height", m_attrs.input_height);
    visitor.on_attribute("input_width", m_attrs.input_width);
    visitor.on_attribute("objectness_score", m_attrs.objectness_score);
    return true;
}
