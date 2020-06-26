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

#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V1 ------------------------------

constexpr NodeTypeInfo op::v1::NonMaxSuppression::type_info;

op::v1::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const op::v1::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
{
    constructor_validate_and_infer_types();
}

op::v1::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const op::v1::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending)
    : Op({boxes,
          scores,
          op::Constant::create(element::i64, Shape{}, {0}),
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::v1::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 5,
                          "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2
                           ? new_args.at(2)
                           : ngraph::op::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3
                           ? new_args.at(3)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4
                           ? new_args.at(4)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v1::NonMaxSuppression>(
        new_args.at(0), new_args.at(1), arg2, arg3, arg4, m_box_encoding, m_sort_result_descending);
}

bool ngraph::op::v1::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    return true;
}

void op::v1::NonMaxSuppression::validate_and_infer_types()
{
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    // the spec doesn't say what exact type should be used for the output of this op
    // that's why we're setting it to 64-bit integer to provide the maximum range of values support
    // this will be changed (configurable) in the next version of this op
    const auto& output_element_type = element::i64;

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    if (boxes_ps.is_dynamic() || scores_ps.is_dynamic())
    {
        set_output_type(0, output_element_type, out_shape);
        return;
    }

    NODE_VALIDATION_CHECK(this,
                          boxes_ps.rank().is_static() && boxes_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'boxes' input. Got: ",
                          boxes_ps);

    NODE_VALIDATION_CHECK(this,
                          scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'scores' input. Got: ",
                          scores_ps);

    if (get_inputs().size() >= 3)
    {
        const auto max_boxes_ps = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              max_boxes_ps.is_dynamic() || is_scalar(max_boxes_ps.to_shape()),
                              "Expected a scalar for the 'max_output_boxes_per_class' input. Got: ",
                              max_boxes_ps);
    }

    if (get_inputs().size() >= 4)
    {
        const auto iou_threshold_ps = get_input_partial_shape(3);
        NODE_VALIDATION_CHECK(this,
                              iou_threshold_ps.is_dynamic() ||
                                  is_scalar(iou_threshold_ps.to_shape()),
                              "Expected a scalar for the 'iou_threshold' input. Got: ",
                              iou_threshold_ps);
    }

    if (get_inputs().size() >= 5)
    {
        const auto score_threshold_ps = get_input_partial_shape(4);
        NODE_VALIDATION_CHECK(this,
                              score_threshold_ps.is_dynamic() ||
                                  is_scalar(score_threshold_ps.to_shape()),
                              "Expected a scalar for the 'score_threshold' input. Got: ",
                              score_threshold_ps);
    }

    const auto num_batches_boxes = boxes_ps[0];
    const auto num_batches_scores = scores_ps[0];
    NODE_VALIDATION_CHECK(this,
                          num_batches_boxes.same_scheme(num_batches_scores),
                          "The first dimension of both 'boxes' and 'scores' must match. Boxes: ",
                          num_batches_boxes,
                          "; Scores: ",
                          num_batches_scores);

    const auto num_boxes_boxes = boxes_ps[1];
    const auto num_boxes_scores = scores_ps[2];
    NODE_VALIDATION_CHECK(this,
                          num_boxes_boxes.same_scheme(num_boxes_scores),
                          "'boxes' and 'scores' input shapes must match at the second and third "
                          "dimension respectively. Boxes: ",
                          num_boxes_boxes,
                          "; Scores: ",
                          num_boxes_scores);

    NODE_VALIDATION_CHECK(this,
                          boxes_ps[2].is_static() && boxes_ps[2].get_length() == 4u,
                          "The last dimension of the 'boxes' input must be equal to 4. Got:",
                          boxes_ps[2]);

    const auto max_output_boxes_per_class = input_value(2).get_node_shared_ptr();
    if (num_boxes_boxes.is_static() && scores_ps[1].is_static() &&
        max_output_boxes_per_class->is_constant())
    {
        const auto num_boxes = num_boxes_boxes.get_length();
        const auto max_output_boxes_per_class = max_boxes_output_from_input();
        const auto num_classes = scores_ps[1].get_length();

        out_shape[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
    }
    set_output_size(1);
    set_output_type(0, output_element_type, out_shape);
}

int64_t op::v1::NonMaxSuppression::max_boxes_output_from_input() const
{
    int64_t max_output_boxes{0};

    const auto max_output_boxes_input =
        as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

namespace ngraph
{
    template <>
    EnumNames<op::v1::NonMaxSuppression::BoxEncodingType>&
        EnumNames<op::v1::NonMaxSuppression::BoxEncodingType>::get()
    {
        static auto enum_names = EnumNames<op::v1::NonMaxSuppression::BoxEncodingType>(
            "op::v1::NonMaxSuppression::BoxEncodingType",
            {{"corner", op::v1::NonMaxSuppression::BoxEncodingType::CORNER},
             {"center", op::v1::NonMaxSuppression::BoxEncodingType::CENTER}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v1::NonMaxSuppression::BoxEncodingType& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

// ------------------------------ V3 ------------------------------

constexpr NodeTypeInfo op::v3::NonMaxSuppression::type_info;

op::v3::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const op::v3::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

op::v3::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const op::v3::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes,
          scores,
          op::Constant::create(element::i64, Shape{}, {0}),
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::v3::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 5,
                          "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2
                           ? new_args.at(2)
                           : ngraph::op::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3
                           ? new_args.at(3)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4
                           ? new_args.at(4)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v3::NonMaxSuppression>(new_args.at(0),
                                                       new_args.at(1),
                                                       arg2,
                                                       arg3,
                                                       arg4,
                                                       m_box_encoding,
                                                       m_sort_result_descending,
                                                       m_output_type);
}

bool ngraph::op::v3::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v3::NonMaxSuppression::validate_inputs()
{
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    if (boxes_ps.is_dynamic() || scores_ps.is_dynamic())
    {
        return;
    }

    NODE_VALIDATION_CHECK(this,
                          boxes_ps.rank().is_static() && boxes_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'boxes' input. Got: ",
                          boxes_ps);

    NODE_VALIDATION_CHECK(this,
                          scores_ps.rank().is_static() && scores_ps.rank().get_length() == 3,
                          "Expected a 3D tensor for the 'scores' input. Got: ",
                          scores_ps);

    if (get_inputs().size() >= 3)
    {
        const auto max_boxes_ps = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              max_boxes_ps.is_dynamic() || is_scalar(max_boxes_ps.to_shape()),
                              "Expected a scalar for the 'max_output_boxes_per_class' input. Got: ",
                              max_boxes_ps);
    }

    if (get_inputs().size() >= 4)
    {
        const auto iou_threshold_ps = get_input_partial_shape(3);
        NODE_VALIDATION_CHECK(this,
                              iou_threshold_ps.is_dynamic() ||
                                  is_scalar(iou_threshold_ps.to_shape()),
                              "Expected a scalar for the 'iou_threshold' input. Got: ",
                              iou_threshold_ps);
    }

    if (get_inputs().size() >= 5)
    {
        const auto score_threshold_ps = get_input_partial_shape(4);
        NODE_VALIDATION_CHECK(this,
                              score_threshold_ps.is_dynamic() ||
                                  is_scalar(score_threshold_ps.to_shape()),
                              "Expected a scalar for the 'score_threshold' input. Got: ",
                              score_threshold_ps);
    }

    const auto num_batches_boxes = boxes_ps[0];
    const auto num_batches_scores = scores_ps[0];
    NODE_VALIDATION_CHECK(this,
                          num_batches_boxes.same_scheme(num_batches_scores),
                          "The first dimension of both 'boxes' and 'scores' must match. Boxes: ",
                          num_batches_boxes,
                          "; Scores: ",
                          num_batches_scores);

    const auto num_boxes_boxes = boxes_ps[1];
    const auto num_boxes_scores = scores_ps[2];
    NODE_VALIDATION_CHECK(this,
                          num_boxes_boxes.same_scheme(num_boxes_scores),
                          "'boxes' and 'scores' input shapes must match at the second and third "
                          "dimension respectively. Boxes: ",
                          num_boxes_boxes,
                          "; Scores: ",
                          num_boxes_scores);

    NODE_VALIDATION_CHECK(this,
                          boxes_ps[2].is_static() && boxes_ps[2].get_length() == 4u,
                          "The last dimension of the 'boxes' input must be equal to 4. Got:",
                          boxes_ps[2]);
}

void op::v3::NonMaxSuppression::validate_and_infer_types()
{
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    if (boxes_ps.is_dynamic() || scores_ps.is_dynamic())
    {
        set_output_type(0, m_output_type, out_shape);
        return;
    }

    validate_inputs();

    const auto num_boxes_boxes = boxes_ps[1];
    const auto max_output_boxes_per_class_node = input_value(2).get_node_shared_ptr();
    if (num_boxes_boxes.is_static() && scores_ps[1].is_static() &&
        max_output_boxes_per_class_node->is_constant())
    {
        const auto num_boxes = num_boxes_boxes.get_length();
        const auto num_classes = scores_ps[1].get_length();
        const auto max_output_boxes_per_class = max_boxes_output_from_input();

        out_shape[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
    }
    set_output_type(0, m_output_type, out_shape);
}

int64_t op::v3::NonMaxSuppression::max_boxes_output_from_input() const
{
    int64_t max_output_boxes{0};

    const auto max_output_boxes_input =
        as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

namespace ngraph
{
    template <>
    EnumNames<op::v3::NonMaxSuppression::BoxEncodingType>&
        EnumNames<op::v3::NonMaxSuppression::BoxEncodingType>::get()
    {
        static auto enum_names = EnumNames<op::v3::NonMaxSuppression::BoxEncodingType>(
            "op::v3::NonMaxSuppression::BoxEncodingType",
            {{"corner", op::v3::NonMaxSuppression::BoxEncodingType::CORNER},
             {"center", op::v3::NonMaxSuppression::BoxEncodingType::CENTER}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v3::NonMaxSuppression::BoxEncodingType>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v3::NonMaxSuppression::BoxEncodingType& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

// ------------------------------ V4 ------------------------------

constexpr NodeTypeInfo op::v4::NonMaxSuppression::type_info;

op::v4::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const op::v4::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : op::v3::NonMaxSuppression(boxes,
                                scores,
                                max_output_boxes_per_class,
                                iou_threshold,
                                score_threshold,
                                box_encoding,
                                sort_result_descending,
                                output_type)
{
    constructor_validate_and_infer_types();
}

op::v4::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const op::v4::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : op::v3::NonMaxSuppression(boxes,
                                scores,
                                op::Constant::create(element::i64, Shape{}, {0}),
                                op::Constant::create(element::f32, Shape{}, {.0f}),
                                op::Constant::create(element::f32, Shape{}, {.0f}),
                                box_encoding,
                                sort_result_descending,
                                output_type)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::v4::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 5,
                          "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2
                           ? new_args.at(2)
                           : ngraph::op::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3
                           ? new_args.at(3)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4
                           ? new_args.at(4)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v4::NonMaxSuppression>(new_args.at(0),
                                                       new_args.at(1),
                                                       arg2,
                                                       arg3,
                                                       arg4,
                                                       m_box_encoding,
                                                       m_sort_result_descending,
                                                       m_output_type);
}

void op::v4::NonMaxSuppression::validate_and_infer_types()
{
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    if (boxes_ps.is_dynamic() || scores_ps.is_dynamic())
    {
        set_output_type(0, m_output_type, out_shape);
        return;
    }

    op::v3::NonMaxSuppression::validate_inputs();

    const auto num_boxes_boxes = boxes_ps[1];
    const auto max_output_boxes_per_class_node = input_value(2).get_node_shared_ptr();
    if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static() &&
        max_output_boxes_per_class_node->is_constant())
    {
        const auto num_boxes = num_boxes_boxes.get_length();
        const auto max_output_boxes_per_class = max_boxes_output_from_input();
        const auto num_classes = scores_ps[1].get_length();

        out_shape[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                       scores_ps[0].get_length();
    }
    set_output_type(0, m_output_type, out_shape);
}

// ------------------------------ dynamic ------------------------------

constexpr NodeTypeInfo op::dynamic::NonMaxSuppression::type_info;

op::dynamic::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const op::dynamic::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : op::v3::NonMaxSuppression(boxes,
                                scores,
                                max_output_boxes_per_class,
                                iou_threshold,
                                score_threshold,
                                box_encoding,
                                sort_result_descending,
                                output_type)
{
    constructor_validate_and_infer_types();
}

op::dynamic::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const op::dynamic::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : op::v3::NonMaxSuppression(boxes,
                                scores,
                                op::Constant::create(element::i64, Shape{}, {0}),
                                op::Constant::create(element::f32, Shape{}, {.0f}),
                                op::Constant::create(element::f32, Shape{}, {.0f}),
                                box_encoding,
                                sort_result_descending,
                                output_type)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::dynamic::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 5,
                          "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2
                           ? new_args.at(2)
                           : ngraph::op::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3
                           ? new_args.at(3)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4
                           ? new_args.at(4)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::dynamic::NonMaxSuppression>(new_args.at(0),
                                                            new_args.at(1),
                                                            arg2,
                                                            arg3,
                                                            arg4,
                                                            m_box_encoding,
                                                            m_sort_result_descending,
                                                            m_output_type);
}

void op::dynamic::NonMaxSuppression::validate_and_infer_types()
{
    op::v3::NonMaxSuppression::validate_and_infer_types();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    set_output_type(0, m_output_type, PartialShape{Dimension::dynamic(), 3});
}
