//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include "itt.hpp"

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
    NGRAPH_OP_SCOPE(v0_DetectionOutput_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this, m_attrs.num_classes > 0, "Number of classes must be greater than zero");

    NODE_VALIDATION_CHECK(
        this, m_attrs.keep_top_k.size() > 0, "keep_top_k attribute must be provided");

    NODE_VALIDATION_CHECK(this,
                          m_attrs.code_type == "caffe.PriorBoxParameter.CORNER" ||
                              m_attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE",
                          "code_type must be either \"caffe.PriorBoxParameter.CORNER\" or "
                          "\"caffe.PriorBoxParameter.CENTER_SIZE\"");

    auto box_logits_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          box_logits_et.is_real(),
                          "Box logits' data type must be floating point. Got " +
                              box_logits_et.get_type_name());
    auto class_preds_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          class_preds_et == box_logits_et,
                          "Class predictions' data type must be the same as box logits type (" +
                              box_logits_et.get_type_name() + "). Got " +
                              class_preds_et.get_type_name());
    auto proposals_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          proposals_et.is_real(),
                          "Proposals' data type must be floating point. Got " +
                              proposals_et.get_type_name());

    const PartialShape& box_logits_pshape = get_input_partial_shape(0);
    const PartialShape& class_preds_pshape = get_input_partial_shape(1);
    const PartialShape& proposals_pshape = get_input_partial_shape(2);

    int num_loc_classes = m_attrs.share_location ? 1 : m_attrs.num_classes;
    int prior_box_size = m_attrs.normalized ? 4 : 5;

    Dimension num_images = Dimension::dynamic();
    Dimension num_prior_boxes = Dimension::dynamic();
    if (box_logits_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              box_logits_pshape.rank().get_length() == 2,
                              "Box logits rank must be 2. Got " +
                                  std::to_string(box_logits_pshape.rank().get_length()));
        num_images = box_logits_pshape[0];
        if (box_logits_pshape[1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                (box_logits_pshape[1].get_length() % (num_loc_classes * 4)) == 0,
                "Box logits' second dimension must be a multiply of num_loc_classes * 4 (" +
                    std::to_string(num_loc_classes * 4) + "). Current value is: ",
                box_logits_pshape[1].get_length(),
                ".");
            num_prior_boxes = box_logits_pshape[1].get_length() / (num_loc_classes * 4);
        }
    }
    if (class_preds_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              class_preds_pshape.rank().get_length() == 2,
                              "Class predictions rank must be 2. Got " +
                                  std::to_string(class_preds_pshape.rank().get_length()));
        if (num_images.is_dynamic() && class_preds_pshape[0].is_static())
        {
            num_images = class_preds_pshape[0];
        }
        else
        {
            NODE_VALIDATION_CHECK(
                this,
                class_preds_pshape[0].compatible(num_images),
                "Class predictions' first dimension is not compatible with batch size.");
        }
        if (class_preds_pshape[1].is_static())
        {
            if (num_prior_boxes.is_dynamic())
            {
                NODE_VALIDATION_CHECK(
                    this,
                    class_preds_pshape[1].get_length() % m_attrs.num_classes == 0,
                    "Class predictions' second dimension must be a multiply of num_classes (" +
                        std::to_string(m_attrs.num_classes) + "). Current value is: ",
                    class_preds_pshape[1].get_length(),
                    ".");
                num_prior_boxes = class_preds_pshape[1].get_length() / m_attrs.num_classes;
            }
            else
            {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(
                    this,
                    class_preds_pshape[1].get_length() == num_prior_boxes_val * m_attrs.num_classes,
                    "Class predictions' second dimension must be equal to num_prior_boxes * "
                    "num_classes (" +
                        std::to_string(num_prior_boxes_val * m_attrs.num_classes) +
                        "). Current value is: ",
                    class_preds_pshape[1].get_length(),
                    ".");
            }
        }
    }
    if (proposals_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              proposals_pshape.rank().get_length() == 3,
                              "Proposals rank must be 3. Got " +
                                  std::to_string(proposals_pshape.rank().get_length()));
        if (num_images.is_static() && proposals_pshape[0].is_static())
        {
            int64_t proposals_1st_dim = proposals_pshape[0].get_length();
            int64_t num_images_val = num_images.get_length();
            NODE_VALIDATION_CHECK(
                this,
                proposals_1st_dim == 1 || proposals_1st_dim == num_images_val,
                "Proposals' first dimension is must be equal to either batch size (" +
                    std::to_string(num_images_val) +
                    ") or 1. Got: " + std::to_string(proposals_1st_dim) + ".");
        }
        if (proposals_pshape[1].is_static())
        {
            size_t proposals_expected_2nd_dim = m_attrs.variance_encoded_in_target ? 1 : 2;
            NODE_VALIDATION_CHECK(this,
                                  proposals_pshape[1].compatible(proposals_expected_2nd_dim),
                                  "Proposals' second dimension is mismatched. Current value is: ",
                                  proposals_pshape[1].get_length(),
                                  ", expected: ",
                                  proposals_expected_2nd_dim,
                                  ".");
        }
        if (proposals_pshape[2].is_static())
        {
            if (num_prior_boxes.is_dynamic())
            {
                NODE_VALIDATION_CHECK(
                    this,
                    proposals_pshape[2].get_length() % prior_box_size == 0,
                    "Proposals' third dimension must be a multiply of prior_box_size (" +
                        std::to_string(prior_box_size) + "). Current value is: ",
                    proposals_pshape[2].get_length(),
                    ".");
                num_prior_boxes = proposals_pshape[2].get_length() / prior_box_size;
            }
            else
            {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(this,
                                      proposals_pshape[2].get_length() ==
                                          num_prior_boxes_val * prior_box_size,
                                      "Proposals' third dimension must be equal to num_prior_boxes "
                                      "* prior_box_size (" +
                                          std::to_string(num_prior_boxes_val * prior_box_size) +
                                          "). Current value is: ",
                                      proposals_pshape[2].get_length(),
                                      ".");
            }
        }
    }

    if (get_input_size() > 3)
    {
        auto aux_class_preds_et = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              aux_class_preds_et == class_preds_et,
                              "Additional class predictions' data type must be the same as class "
                              "predictions data type (" +
                                  class_preds_et.get_type_name() + "). Got " +
                                  aux_class_preds_et.get_type_name());
        auto aux_box_preds_et = get_input_element_type(4);
        NODE_VALIDATION_CHECK(
            this,
            aux_box_preds_et == box_logits_et,
            "Additional box predictions' data type must be the same as box logits data type (" +
                box_logits_et.get_type_name() + "). Got " + aux_box_preds_et.get_type_name());

        const PartialShape& aux_class_preds_pshape = get_input_partial_shape(3);
        const PartialShape& aux_box_preds_pshape = get_input_partial_shape(4);
        if (aux_class_preds_pshape.rank().is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  aux_class_preds_pshape[0].compatible(num_images),
                                  "Additional class predictions' first dimension must be "
                                  "compatible with batch size.");
            if (num_prior_boxes.is_static())
            {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(
                    this,
                    aux_class_preds_pshape[1].get_length() == num_prior_boxes_val * 2,
                    "Additional class predictions' second dimension must be equal to "
                    "num_prior_boxes * 2 (" +
                        std::to_string(num_prior_boxes_val * 2) + "). Got " +
                        std::to_string(aux_class_preds_pshape[1].get_length()) + ".");
            }
        }
        NODE_VALIDATION_CHECK(
            this,
            aux_box_preds_pshape.compatible(box_logits_pshape),
            "Additional box predictions' shape must be compatible with box logits shape.");
    }

    std::vector<Dimension> output_shape{1, 1};
    if (m_attrs.keep_top_k[0] > 0)
    {
        output_shape.push_back(num_images * m_attrs.keep_top_k[0]);
    }
    else if (m_attrs.top_k > 0)
    {
        output_shape.push_back(num_images * m_attrs.top_k * m_attrs.num_classes);
    }
    else
    {
        output_shape.push_back(num_images * num_prior_boxes * m_attrs.num_classes);
    }
    output_shape.push_back(7);

    set_output_type(0, box_logits_et, output_shape);
}

shared_ptr<Node> op::DetectionOutput::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_DetectionOutput_clone_with_new_inputs);
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
    NGRAPH_OP_SCOPE(v0_DetectionOutput_visit_attributes);
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
