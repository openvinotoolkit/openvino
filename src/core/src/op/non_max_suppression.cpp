// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_max_suppression.hpp"

#include <cstring>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "nms_shape_inference.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov {
// ------------------------------ V1 ------------------------------

op::v1::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const op::v1::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending} {
    constructor_validate_and_infer_types();
}

op::v1::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const op::v1::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending)
    : Op({boxes,
          scores,
          op::v0::Constant::create(element::i64, Shape{}, {0}),
          op::v0::Constant::create(element::f32, Shape{}, {.0f}),
          op::v0::Constant::create(element::f32, Shape{}, {.0f})}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v1::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_NonMaxSuppression_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2 && new_args.size() <= 5, "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : op::v0::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : op::v0::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : op::v0::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v1::NonMaxSuppression>(new_args.at(0),
                                                       new_args.at(1),
                                                       arg2,
                                                       arg3,
                                                       arg4,
                                                       m_box_encoding,
                                                       m_sort_result_descending);
}

bool op::v1::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_NonMaxSuppression_visit_attributes);
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    return true;
}

void op::v1::NonMaxSuppression::validate_and_infer_types() {
    OV_OP_SCOPE(v1_NonMaxSuppression_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, element::i64, output_shapes.front());
}

int64_t op::v1::NonMaxSuppression::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    const auto max_output_boxes_input = ov::util::get_constant_from_source(input_value(2));
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

template <>
OPENVINO_API EnumNames<op::v1::NonMaxSuppression::BoxEncodingType>&
EnumNames<op::v1::NonMaxSuppression::BoxEncodingType>::get() {
    static auto enum_names = EnumNames<op::v1::NonMaxSuppression::BoxEncodingType>(
        "op::v1::NonMaxSuppression::BoxEncodingType",
        {{"corner", op::v1::NonMaxSuppression::BoxEncodingType::CORNER},
         {"center", op::v1::NonMaxSuppression::BoxEncodingType::CENTER}});
    return enum_names;
}

std::ostream& operator<<(std::ostream& s, const op::v1::NonMaxSuppression::BoxEncodingType& type) {
    return s << as_string(type);
}

AttributeAdapter<op::v1::NonMaxSuppression::BoxEncodingType>::~AttributeAdapter() = default;

// ------------------------------ V3 ------------------------------
op::v3::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const op::v3::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v3::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const op::v3::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes,
          scores,
          op::v0::Constant::create(element::i64, Shape{}, {0}),
          op::v0::Constant::create(element::f32, Shape{}, {.0f}),
          op::v0::Constant::create(element::f32, Shape{}, {.0f})}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v3::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_NonMaxSuppression_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2 && new_args.size() <= 5, "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : op::v0::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : op::v0::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : op::v0::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v3::NonMaxSuppression>(new_args.at(0),
                                                       new_args.at(1),
                                                       arg2,
                                                       arg3,
                                                       arg4,
                                                       m_box_encoding,
                                                       m_sort_result_descending,
                                                       m_output_type);
}

bool op::v3::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_NonMaxSuppression_visit_attributes);
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v3::NonMaxSuppression::validate_and_infer_types() {
    OV_OP_SCOPE(v3_NonMaxSuppression_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, m_output_type, output_shapes.front());
}

int64_t op::v3::NonMaxSuppression::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    const auto max_output_boxes_input = ov::util::get_constant_from_source(input_value(2));
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

template <>
OPENVINO_API EnumNames<op::v3::NonMaxSuppression::BoxEncodingType>&
EnumNames<op::v3::NonMaxSuppression::BoxEncodingType>::get() {
    static auto enum_names = EnumNames<op::v3::NonMaxSuppression::BoxEncodingType>(
        "op::v3::NonMaxSuppression::BoxEncodingType",
        {{"corner", op::v3::NonMaxSuppression::BoxEncodingType::CORNER},
         {"center", op::v3::NonMaxSuppression::BoxEncodingType::CENTER}});
    return enum_names;
}

std::ostream& operator<<(std::ostream& s, const op::v3::NonMaxSuppression::BoxEncodingType& type) {
    return s << as_string(type);
}

AttributeAdapter<op::v3::NonMaxSuppression::BoxEncodingType>::~AttributeAdapter() = default;

// ------------------------------ V4 ------------------------------
op::v4::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
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
                                output_type) {
    constructor_validate_and_infer_types();
}

op::v4::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const op::v4::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : op::v3::NonMaxSuppression(boxes,
                                scores,
                                op::v0::Constant::create(element::i64, Shape{}, {0}),
                                op::v0::Constant::create(element::f32, Shape{}, {.0f}),
                                op::v0::Constant::create(element::f32, Shape{}, {.0f}),
                                box_encoding,
                                sort_result_descending,
                                output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v4::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_NonMaxSuppression_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2 && new_args.size() <= 5, "Number of inputs must be 2, 3, 4 or 5");

    const auto& arg2 = new_args.size() > 2 ? new_args.at(2) : op::v0::Constant::create(element::i32, Shape{}, {0});
    const auto& arg3 = new_args.size() > 3 ? new_args.at(3) : op::v0::Constant::create(element::f32, Shape{}, {.0f});
    const auto& arg4 = new_args.size() > 4 ? new_args.at(4) : op::v0::Constant::create(element::f32, Shape{}, {.0f});

    return std::make_shared<op::v4::NonMaxSuppression>(new_args.at(0),
                                                       new_args.at(1),
                                                       arg2,
                                                       arg3,
                                                       arg4,
                                                       m_box_encoding,
                                                       m_sort_result_descending,
                                                       m_output_type);
}

void op::v4::NonMaxSuppression::validate_and_infer_types() {
    OV_OP_SCOPE(v4_NonMaxSuppression_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, m_output_type, output_shapes.front());
}

// ------------------------------ V5 ------------------------------
op::v5::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v5::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const Output<Node>& soft_nms_sigma,
                                             const op::v5::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v5::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_NonMaxSuppression_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 6,
                          "Number of inputs must be 2, 3, 4, 5 or 6");

    switch (new_args.size()) {
    case 2:
        return std::make_shared<op::v5::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 3:
        return std::make_shared<op::v5::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 4:
        return std::make_shared<op::v5::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 5:
        return std::make_shared<op::v5::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    default:
        return std::make_shared<op::v5::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           new_args.at(5),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    }
}

namespace {
constexpr size_t max_output_boxes_port = 2;
constexpr size_t iou_threshold_port = 3;
constexpr size_t score_threshold_port = 4;
constexpr size_t soft_nms_sigma_port = 5;

inline bool is_float_type_admissible(const element::Type& t) {
    return t == element::dynamic || t == element::f32 || t == element::f16 || t == element::bf16;
}
}  // namespace

namespace op {
namespace nms {
namespace validate {
namespace {
void input_types(const Node* op) {
    NODE_VALIDATION_CHECK(op,
                          is_float_type_admissible(op->get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(op,
                          is_float_type_admissible(op->get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");
    const auto inputs_size = op->get_input_size();
    if (inputs_size > 3) {
        NODE_VALIDATION_CHECK(op,
                              is_float_type_admissible(op->get_input_element_type(3)),
                              "Expected bf16, fp16 or fp32 as element type for the "
                              "'iou_threshold' input.");
    }

    if (inputs_size > 4) {
        NODE_VALIDATION_CHECK(op,
                              is_float_type_admissible(op->get_input_element_type(4)),
                              "Expected bf16, fp16 or fp32 as element type for the "
                              "'score_threshold_ps' input.");
    }

    if (inputs_size > 5) {
        NODE_VALIDATION_CHECK(op,
                              is_float_type_admissible(op->get_input_element_type(5)),
                              "Expected bf16, fp16 or fp32 as element type for the "
                              "'soft_nms_sigma' input.");
    }
}
}  // namespace
}  // namespace validate
}  // namespace nms
}  // namespace op

int64_t op::v5::NonMaxSuppression::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    if (inputs().size() < 3) {
        return 0;
    }

    const auto max_output_boxes_input = ov::util::get_constant_from_source(input_value(max_output_boxes_port));
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

float op::v5::NonMaxSuppression::iou_threshold_from_input() const {
    float iou_threshold = 0.0f;

    if (inputs().size() < 4) {
        return iou_threshold;
    }

    const auto iou_threshold_input = ov::util::get_constant_from_source(input_value(iou_threshold_port));
    iou_threshold = iou_threshold_input->cast_vector<float>().at(0);

    return iou_threshold;
}

float op::v5::NonMaxSuppression::score_threshold_from_input() const {
    float score_threshold = 0.0f;

    if (inputs().size() < 5) {
        return score_threshold;
    }

    const auto score_threshold_input = ov::util::get_constant_from_source(input_value(score_threshold_port));
    score_threshold = score_threshold_input->cast_vector<float>().at(0);

    return score_threshold;
}

float op::v5::NonMaxSuppression::soft_nms_sigma_from_input() const {
    float soft_nms_sigma = 0.0f;

    if (inputs().size() < 6) {
        return soft_nms_sigma;
    }

    const auto soft_nms_sigma_input = ov::util::get_constant_from_source(input_value(soft_nms_sigma_port));
    soft_nms_sigma = soft_nms_sigma_input->cast_vector<float>().at(0);

    return soft_nms_sigma;
}

bool op::v5::NonMaxSuppression::is_soft_nms_sigma_constant_and_default() const {
    auto soft_nms_sigma_node = input_value(soft_nms_sigma_port).get_node_shared_ptr();
    if (inputs().size() < 6 || !op::util::is_constant(soft_nms_sigma_node)) {
        return false;
    }
    const auto soft_nms_sigma_input = as_type_ptr<op::v0::Constant>(soft_nms_sigma_node);
    return soft_nms_sigma_input->cast_vector<float>().at(0) == 0.0f;
}

bool op::v5::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_NonMaxSuppression_visit_attributes);
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v5::NonMaxSuppression::validate_and_infer_types() {
    OV_OP_SCOPE(v5_NonMaxSuppression_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    nms::validate::input_types(this);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    set_output_type(0, m_output_type, output_shapes[0]);
    set_output_type(1, element::f32, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}

std::ostream& operator<<(std::ostream& s, const op::v5::NonMaxSuppression::BoxEncodingType& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v5::NonMaxSuppression::BoxEncodingType>&
EnumNames<op::v5::NonMaxSuppression::BoxEncodingType>::get() {
    static auto enum_names = EnumNames<op::v5::NonMaxSuppression::BoxEncodingType>(
        "op::v5::NonMaxSuppression::BoxEncodingType",
        {{"corner", op::v5::NonMaxSuppression::BoxEncodingType::CORNER},
         {"center", op::v5::NonMaxSuppression::BoxEncodingType::CENTER}});
    return enum_names;
}

// ------------------------------ V9 ------------------------------
op::v9::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const op::v9::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v9::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const op::v9::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v9::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const op::v9::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v9::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const op::v9::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

op::v9::NonMaxSuppression::NonMaxSuppression(const Output<Node>& boxes,
                                             const Output<Node>& scores,
                                             const Output<Node>& max_output_boxes_per_class,
                                             const Output<Node>& iou_threshold,
                                             const Output<Node>& score_threshold,
                                             const Output<Node>& soft_nms_sigma,
                                             const op::v9::NonMaxSuppression::BoxEncodingType box_encoding,
                                             const bool sort_result_descending,
                                             const element::Type& output_type)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, soft_nms_sigma}),
      m_box_encoding{box_encoding},
      m_sort_result_descending{sort_result_descending},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v9::NonMaxSuppression::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_NonMaxSuppression_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() >= 2 && new_args.size() <= 6,
                          "Number of inputs must be 2, 3, 4, 5 or 6");

    switch (new_args.size()) {
    case 2:
        return std::make_shared<op::v9::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 3:
        return std::make_shared<op::v9::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 4:
        return std::make_shared<op::v9::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    case 5:
        return std::make_shared<op::v9::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    default:
        return std::make_shared<op::v9::NonMaxSuppression>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           new_args.at(5),
                                                           m_box_encoding,
                                                           m_sort_result_descending,
                                                           m_output_type);
        break;
    }
}

int64_t op::v9::NonMaxSuppression::max_boxes_output_from_input() const {
    int64_t max_output_boxes{0};

    if (inputs().size() < 3) {
        return 0;
    }

    const auto max_output_boxes_input = ov::util::get_constant_from_source(input_value(max_output_boxes_port));
    max_output_boxes = max_output_boxes_input->cast_vector<int64_t>().at(0);

    return max_output_boxes;
}

float op::v9::NonMaxSuppression::iou_threshold_from_input() const {
    float iou_threshold = 0.0f;

    if (inputs().size() < 4) {
        return iou_threshold;
    }

    const auto iou_threshold_input = ov::util::get_constant_from_source(input_value(iou_threshold_port));
    iou_threshold = iou_threshold_input->cast_vector<float>().at(0);

    return iou_threshold;
}

float op::v9::NonMaxSuppression::score_threshold_from_input() const {
    float score_threshold = 0.0f;

    if (inputs().size() < 5) {
        return score_threshold;
    }

    const auto score_threshold_input = ov::util::get_constant_from_source(input_value(score_threshold_port));
    score_threshold = score_threshold_input->cast_vector<float>().at(0);

    return score_threshold;
}

float op::v9::NonMaxSuppression::soft_nms_sigma_from_input() const {
    float soft_nms_sigma = 0.0f;

    if (inputs().size() < 6) {
        return soft_nms_sigma;
    }

    const auto soft_nms_sigma_input = ov::util::get_constant_from_source(input_value(soft_nms_sigma_port));
    soft_nms_sigma = soft_nms_sigma_input->cast_vector<float>().at(0);

    return soft_nms_sigma;
}

bool op::v9::NonMaxSuppression::is_soft_nms_sigma_constant_and_default() const {
    auto soft_nms_sigma_node = input_value(soft_nms_sigma_port).get_node_shared_ptr();
    if (inputs().size() < 6 || !op::util::is_constant(soft_nms_sigma_node)) {
        return false;
    }
    const auto soft_nms_sigma_input = as_type_ptr<op::v0::Constant>(soft_nms_sigma_node);
    return soft_nms_sigma_input->cast_vector<float>().at(0) == 0.0f;
}

bool op::v9::NonMaxSuppression::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_NonMaxSuppression_visit_attributes);
    visitor.on_attribute("box_encoding", m_box_encoding);
    visitor.on_attribute("sort_result_descending", m_sort_result_descending);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void op::v9::NonMaxSuppression::validate_and_infer_types() {
    OV_OP_SCOPE(v9_NonMaxSuppression_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    const auto output_shapes = shape_infer(this, input_shapes);

    nms::validate::input_types(this);
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");

    set_output_type(0, m_output_type, output_shapes[0]);
    set_output_type(1, element::f32, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}

std::ostream& operator<<(std::ostream& s, const op::v9::NonMaxSuppression::BoxEncodingType& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v9::NonMaxSuppression::BoxEncodingType>&
EnumNames<op::v9::NonMaxSuppression::BoxEncodingType>::get() {
    static auto enum_names = EnumNames<op::v9::NonMaxSuppression::BoxEncodingType>(
        "op::v9::NonMaxSuppression::BoxEncodingType",
        {{"corner", op::v9::NonMaxSuppression::BoxEncodingType::CORNER},
         {"center", op::v9::NonMaxSuppression::BoxEncodingType::CENTER}});
    return enum_names;
}

AttributeAdapter<op::v5::NonMaxSuppression::BoxEncodingType>::~AttributeAdapter() = default;
AttributeAdapter<op::v9::NonMaxSuppression::BoxEncodingType>::~AttributeAdapter() = default;
}  // namespace ov
