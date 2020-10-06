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
#include <cstring>
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/non_max_suppression.hpp"
#include "ngraph/type/float16.hpp"

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

    if (inputs().size() >= 3)
    {
        const auto max_boxes_ps = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              max_boxes_ps.is_dynamic() || is_scalar(max_boxes_ps.to_shape()),
                              "Expected a scalar for the 'max_output_boxes_per_class' input. Got: ",
                              max_boxes_ps);
    }

    if (inputs().size() >= 4)
    {
        const auto iou_threshold_ps = get_input_partial_shape(3);
        NODE_VALIDATION_CHECK(this,
                              iou_threshold_ps.is_dynamic() ||
                                  is_scalar(iou_threshold_ps.to_shape()),
                              "Expected a scalar for the 'iou_threshold' input. Got: ",
                              iou_threshold_ps);
    }

    if (inputs().size() >= 5)
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
        op::is_constant(max_output_boxes_per_class))
    {
        const auto num_boxes = num_boxes_boxes.get_length();
        const auto max_output_boxes_per_class = max_boxes_output_from_input();
        const auto num_classes = scores_ps[1].get_length();

        out_shape[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
    }
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

void op::v3::NonMaxSuppression::validate()
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

    if (inputs().size() >= 3)
    {
        const auto max_boxes_ps = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              max_boxes_ps.is_dynamic() || is_scalar(max_boxes_ps.to_shape()),
                              "Expected a scalar for the 'max_output_boxes_per_class' input. Got: ",
                              max_boxes_ps);
    }

    if (inputs().size() >= 4)
    {
        const auto iou_threshold_ps = get_input_partial_shape(3);
        NODE_VALIDATION_CHECK(this,
                              iou_threshold_ps.is_dynamic() ||
                                  is_scalar(iou_threshold_ps.to_shape()),
                              "Expected a scalar for the 'iou_threshold' input. Got: ",
                              iou_threshold_ps);
    }

    if (inputs().size() >= 5)
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

    validate();

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
    {
        const auto num_boxes_boxes = boxes_ps[1];
        const auto max_output_boxes_per_class_node = input_value(2).get_node_shared_ptr();
        if (num_boxes_boxes.is_static() && scores_ps[1].is_static() &&
            op::is_constant(max_output_boxes_per_class_node))
        {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            const auto max_output_boxes_per_class = max_boxes_output_from_input();

            out_shape[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
        }
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

    op::v3::NonMaxSuppression::validate();

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
    {
        const auto num_boxes_boxes = boxes_ps[1];
        const auto max_output_boxes_per_class_node = input_value(2).get_node_shared_ptr();
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static() &&
            op::is_constant(max_output_boxes_per_class_node))
        {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            const auto max_output_boxes_per_class = max_boxes_output_from_input();

            out_shape[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                           scores_ps[0].get_length();
        }
    }
    set_output_type(0, m_output_type, out_shape);
}

// ------------------------------ V5 ------------------------------

constexpr NodeTypeInfo op::v5::NonMaxSuppression::type_info;

op::v5::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes,
          scores,
          op::Constant::create(element::i64, Shape{}, {0}),
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes,
          scores,
          max_output_boxes_per_class,
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes,
          scores,
          max_output_boxes_per_class,
          iou_threshold,
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes,
          scores,
          max_output_boxes_per_class,
          iou_threshold,
          score_threshold,
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const Output<Node>& soft_nms_sigma,
    const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending,
    const element::Type& output_type)
    : Op({boxes,
          scores,
          max_output_boxes_per_class,
          iou_threshold,
          score_threshold,
          soft_nms_sigma})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
    , m_output_type{output_type}
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::v5::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 6,
                          "Number of inputs must be 2, 3, 4, 5 or 6");

    const auto& arg2 = new_args.size() > 2
                           ? new_args.at(2)
                           : ngraph::op::Constant::create(element::i64, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3
                           ? new_args.at(3)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4
                           ? new_args.at(4)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg5 = new_args.size() > 5
                           ? new_args.at(5)
                           : ngraph::op::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v5::NonMaxSuppression>(new_args.at(0),
                                                       new_args.at(1),
                                                       arg2,
                                                       arg3,
                                                       arg4,
                                                       arg5,
                                                       m_box_encoding,
                                                       m_sort_result_descending,
                                                       m_output_type);
}

void op::v5::NonMaxSuppression::validate()
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

    if (inputs().size() >= 3)
    {
        const auto max_boxes_ps = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              max_boxes_ps.is_dynamic() || is_scalar(max_boxes_ps.to_shape()),
                              "Expected a scalar for the 'max_output_boxes_per_class' input. Got: ",
                              max_boxes_ps);
    }

    if (inputs().size() >= 4)
    {
        const auto iou_threshold_ps = get_input_partial_shape(3);
        NODE_VALIDATION_CHECK(this,
                              iou_threshold_ps.is_dynamic() ||
                                  is_scalar(iou_threshold_ps.to_shape()),
                              "Expected a scalar for the 'iou_threshold' input. Got: ",
                              iou_threshold_ps);
    }

    if (inputs().size() >= 5)
    {
        const auto score_threshold_ps = get_input_partial_shape(4);
        NODE_VALIDATION_CHECK(this,
                              score_threshold_ps.is_dynamic() ||
                                  is_scalar(score_threshold_ps.to_shape()),
                              "Expected a scalar for the 'score_threshold' input. Got: ",
                              score_threshold_ps);
    }

    if (inputs().size() >= 6)
    {
        const auto soft_nms_sigma = get_input_partial_shape(5);
        NODE_VALIDATION_CHECK(this,
                              soft_nms_sigma.is_dynamic() || is_scalar(soft_nms_sigma.to_shape()),
                              "Expected a scalar for the 'soft_nms_sigma' input. Got: ",
                              soft_nms_sigma);
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

int64_t op::v5::NonMaxSuppression::max_boxes_output_from_input() const
{
    int64_t max_output_boxes{0};

    const auto max_output_boxes_input =
        as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

static constexpr size_t boxes_port = 0;
static constexpr size_t scores_port = 1;
static constexpr size_t iou_threshold_port = 3;
static constexpr size_t score_threshold_port = 4;
static constexpr size_t soft_nms_sigma_port = 5;

float op::v5::NonMaxSuppression::iou_threshold_from_input() const
{
    float iou_threshold = 0.0f;

    const auto iou_threshold_input =
        as_type_ptr<op::Constant>(input_value(iou_threshold_port).get_node_shared_ptr());
    iou_threshold = iou_threshold_input->cast_vector<float>().at(0);

    return iou_threshold;
}

float op::v5::NonMaxSuppression::score_threshold_from_input() const
{
    float score_threshold = 0.0f;

    const auto score_threshold_input =
        as_type_ptr<op::Constant>(input_value(score_threshold_port).get_node_shared_ptr());
    score_threshold = score_threshold_input->cast_vector<float>().at(0);

    return score_threshold;
}

float op::v5::NonMaxSuppression::soft_nms_sigma_from_input() const
{
    float soft_nms_sigma = 0.0f;

    const auto soft_nms_sigma_input =
        as_type_ptr<op::Constant>(input_value(soft_nms_sigma_port).get_node_shared_ptr());
    soft_nms_sigma = soft_nms_sigma_input->cast_vector<float>().at(0);

    return soft_nms_sigma;
}

bool ngraph::op::v5::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v5::NonMaxSuppression::validate_and_infer_types()
{
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    validate();

    set_output_type(0, m_output_type, out_shape);
    set_output_type(1, element::f32, out_shape);
    set_output_type(2, m_output_type, Shape{});
}

namespace ngraph
{
    template <>
    EnumNames<op::v5::NonMaxSuppression::BoxEncodingType>&
        EnumNames<op::v5::NonMaxSuppression::BoxEncodingType>::get()
    {
        static auto enum_names = EnumNames<op::v5::NonMaxSuppression::BoxEncodingType>(
            "op::v5::NonMaxSuppression::BoxEncodingType",
            {{"corner", op::v5::NonMaxSuppression::BoxEncodingType::CORNER},
             {"center", op::v5::NonMaxSuppression::BoxEncodingType::CENTER}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo
        AttributeAdapter<op::v5::NonMaxSuppression::BoxEncodingType>::type_info;

    std::ostream& operator<<(std::ostream& s,
                             const op::v5::NonMaxSuppression::BoxEncodingType& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

static PartialShape infer_selected_indices_shape(const HostTensorVector& inputs,
                                                 int64_t max_output_boxes_per_class)
{
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
    {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static())
        {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                        scores_ps[0].get_length();
        }
    }

    return result;
}

using V5BoxEncoding = op::v5::NonMaxSuppression::BoxEncodingType;

static void normalize_corner(float* boxes, const Shape& boxes_shape)
{
    size_t num_batches = boxes_shape[0];
    size_t num_boxes = boxes_shape[1];

    float* box_ptr = boxes;

    for (size_t batch = 0; batch < num_batches; ++batch)
    {
        for (size_t box = 0; box < num_boxes; ++num_boxes)
        {
            float y1 = box_ptr[0];
            float x1 = box_ptr[1];
            float y2 = box_ptr[2];
            float x2 = box_ptr[3];

            float ymin = std::min(y1, y2);
            float ymax = std::max(y1, y2);
            float xmin = std::min(x1, x2);
            float xmax = std::max(x1, x2);

            box_ptr[0] = ymin;
            box_ptr[1] = xmin;
            box_ptr[2] = ymax;
            box_ptr[3] = xmax;

            box_ptr += 4;
        }
    }
}

static void normalize_center(float* boxes, const Shape& boxes_shape)
{
    size_t num_batches = boxes_shape[0];
    size_t num_boxes = boxes_shape[1];

    float* box_ptr = boxes;

    for (size_t batch = 0; batch < num_batches; ++batch)
    {
        for (size_t box = 0; box < num_boxes; ++num_boxes)
        {
            float x_center = box_ptr[0];
            float y_center = box_ptr[1];
            float width = box_ptr[2];
            float height = box_ptr[3];

            float y1 = y_center - height / 2.0;
            float x1 = x_center - width / 2.0;
            float y2 = y_center + height / 2.0;
            float x2 = x_center + width / 2.0;

            box_ptr[0] = y1;
            box_ptr[1] = x1;
            box_ptr[2] = y2;
            box_ptr[3] = x2;

            box_ptr += 4;
        }
    }
}

static void
    normalize_box_encoding(float* boxes, const Shape& boxes_shape, const V5BoxEncoding box_encoding)
{
    if (box_encoding == V5BoxEncoding::CORNER)
    {
        normalize_corner(boxes, boxes_shape);
    }
    else
    {
        normalize_center(boxes, boxes_shape);
    }
}

static std::vector<float> prepare_boxes_data(const HostTensorPtr& boxes,
                                             const Shape& boxes_shape,
                                             const V5BoxEncoding box_encoding)
{
    element::Type boxes_input_et = boxes->get_element_type();

    size_t boxes_size = shape_size(boxes_shape);
    std::vector<float> result(boxes_size);

    if (boxes_input_et == ngraph::element::f32)
    {
        float* boxes_ptr = boxes->get_data_ptr<float>();
        memcpy(result.data(), boxes_ptr, boxes_size * sizeof(float));
    }
    else
    {
        float16* boxes_ptr = boxes->get_data_ptr<float16>();
        for (size_t i = 0; i < boxes_size; ++i)
        {
            result[i] = float(boxes_ptr[i]);
        }
    }

    normalize_box_encoding(result.data(), boxes_shape, box_encoding);

    return result;
}

static std::vector<float> prepare_scores_data(const HostTensorPtr& scores,
                                              const Shape& scores_shape)
{
    element::Type scores_input_et = scores->get_element_type();

    size_t scores_size = shape_size(scores_shape);
    std::vector<float> result(scores_size);

    if (scores_input_et == ngraph::element::f32)
    {
        float* scores_ptr = scores->get_data_ptr<float>();
        memcpy(result.data(), scores_ptr, scores_size * sizeof(float));
    }
    else
    {
        float16* scores_ptr = scores->get_data_ptr<float16>();
        for (size_t i = 0; i < scores_size; ++i)
        {
            result[i] = float(scores_ptr[i]);
        }
    }

    return result;
}

static void evaluate_postprocessing(const HostTensorVector& outputs,
                                    const ngraph::element::Type output_type,
                                    const std::vector<int64_t>& selected_indices,
                                    const std::vector<float>& selected_scores,
                                    int64_t valid_outputs)
{
    size_t num_of_outputs = outputs.size();
    size_t selected_size = valid_outputs * 3;

    if (output_type == ngraph::element::i64)
    {
        int64_t* indices_ptr = outputs[0]->get_data_ptr<int64_t>();
        memcpy(indices_ptr, selected_indices.data(), selected_size * sizeof(int64_t));
    }
    else
    {
        int32_t* indices_ptr = outputs[0]->get_data_ptr<int32_t>();
        for (size_t i = 0; i < selected_size; ++i)
        {
            indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
        }
    }

    if (num_of_outputs < 2)
    {
        return;
    }

    float* scores_ptr = outputs[1]->get_data_ptr<float>();
    memcpy(scores_ptr, selected_scores.data(), selected_size * sizeof(float));

    if (num_of_outputs < 3)
    {
        return;
    }

    if (output_type == ngraph::element::i64)
    {
        int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
        *valid_outputs_ptr = valid_outputs;
    }
    else
    {
        int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
        *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
    }
}

bool op::v5::NonMaxSuppression::evaluate(const HostTensorVector& outputs,
                                         const HostTensorVector& inputs) const
{
    int64_t max_output_boxes_per_class = max_boxes_output_from_input();
    float iou_threshold = iou_threshold_from_input();
    float score_threshold = score_threshold_from_input();
    float soft_nms_sigma = soft_nms_sigma_from_input();

    auto selected_indices_shape = infer_selected_indices_shape(inputs, max_output_boxes_per_class);
    Shape out_shape = selected_indices_shape.to_shape();

    Shape boxes_shape = inputs[boxes_port]->get_shape();
    Shape scores_shape = inputs[scores_port]->get_shape();

    auto boxes_data = prepare_boxes_data(inputs[boxes_port], boxes_shape, m_box_encoding);
    auto scores_data = prepare_scores_data(inputs[scores_port], scores_shape);

    size_t out_shape_size = shape_size(out_shape);

    std::vector<int64_t> selected_indices(out_shape_size);
    std::vector<float> selected_scores(out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression(boxes_data.data(),
                                            boxes_shape,
                                            scores_data.data(),
                                            scores_shape,
                                            max_output_boxes_per_class,
                                            iou_threshold,
                                            score_threshold,
                                            soft_nms_sigma,
                                            selected_indices.data(),
                                            out_shape,
                                            selected_scores.data(),
                                            out_shape,
                                            &valid_outputs,
                                            m_sort_result_descending);

    outputs[0]->set_element_type(m_output_type);
    outputs[0]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});

    size_t num_of_outputs = outputs.size();

    if (num_of_outputs >= 2)
    {
        outputs[1]->set_element_type(element::f32);
        outputs[1]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});
    }

    if (num_of_outputs >= 3)
    {
        outputs[2]->set_element_type(m_output_type);
        outputs[2]->set_shape(Shape{});
    }

    evaluate_postprocessing(
        outputs, m_output_type, selected_indices, selected_scores, valid_outputs);

    return true;
}
