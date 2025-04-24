// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/prior_box.hpp"

#include <array>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/op/prior_box.hpp"
#include "openvino/runtime/tensor.hpp"
#include "prior_box_shape_inference.hpp"

namespace ov {
namespace op {
// ------------------------------ V0 ------------------------------
namespace v0 {
namespace {
template <element::Type_t ET>
bool evaluate(const Tensor& arg0, const Tensor& arg1, Tensor& out, const op::v0::PriorBox::Attributes& attrs) {
    op::v8::PriorBox::Attributes attrs_v8;
    attrs_v8.min_size = attrs.min_size;
    attrs_v8.max_size = attrs.max_size;
    attrs_v8.aspect_ratio = attrs.aspect_ratio;
    attrs_v8.density = attrs.density;
    attrs_v8.fixed_ratio = attrs.fixed_ratio;
    attrs_v8.fixed_size = attrs.fixed_size;
    attrs_v8.clip = attrs.clip;
    attrs_v8.flip = attrs.flip;
    attrs_v8.step = attrs.step;
    attrs_v8.offset = attrs.offset;
    attrs_v8.variance = attrs.variance;
    attrs_v8.scale_all_sizes = attrs.scale_all_sizes;

    using T = typename element_type_traits<ET>::value_type;
    ov::reference::prior_box(arg0.data<T>(), arg1.data<T>(), out.data<float>(), out.get_shape(), attrs_v8);
    return true;
}

bool evaluate_prior_box(const Tensor& arg0,
                        const Tensor& arg1,
                        Tensor& out,
                        const op::v0::PriorBox::Attributes& attrs) {
    bool rc = true;
    switch (arg0.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

PriorBox::PriorBox(const Output<Node>& layer_shape, const Output<Node>& image_shape, const PriorBox::Attributes& attrs)
    : Op({layer_shape, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void PriorBox::validate_and_infer_types() {
    OV_OP_SCOPE(v0_PriorBox_validate_and_infer_types);

    const auto input_shapes = prior_box::validate::inputs_et(this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, element::f32, output_shapes.front());
    set_input_is_relevant_to_shape(0);
}

std::shared_ptr<Node> PriorBox::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PriorBox_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PriorBox>(new_args.at(0), new_args.at(1), m_attrs);
}

int64_t op::v0::PriorBox::number_of_priors(const PriorBox::Attributes& attrs) {
    // Starting with 0 number of prior and then various conditions on attributes will contribute
    // real number of prior boxes as PriorBox is a fat thing with several modes of
    // operation that will be checked in order in the next statements.
    int64_t num_priors = 0;

    // Total number of boxes around each point; depends on whether flipped boxes are included
    // plus one box 1x1.
    int64_t total_aspect_ratios = normalized_aspect_ratio(attrs.aspect_ratio, attrs.flip).size();

    if (attrs.scale_all_sizes)
        num_priors = total_aspect_ratios * attrs.min_size.size() + attrs.max_size.size();
    else
        num_priors = total_aspect_ratios + attrs.min_size.size() - 1;

    if (!attrs.fixed_size.empty())
        num_priors = total_aspect_ratios * attrs.fixed_size.size();

    for (auto density : attrs.density) {
        auto rounded_density = static_cast<int64_t>(density);
        auto density_2d = (rounded_density * rounded_density - 1);
        if (!attrs.fixed_ratio.empty())
            num_priors += attrs.fixed_ratio.size() * density_2d;
        else
            num_priors += total_aspect_ratios * density_2d;
    }
    return num_priors;
}

std::vector<float> PriorBox::normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip) {
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio) {
        unique_ratios.insert(std::round(ratio * 1e6f) / 1e6f);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6f) / 1e6f);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

bool PriorBox::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_PriorBox_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("max_size", m_attrs.max_size);
    visitor.on_attribute("aspect_ratio", m_attrs.aspect_ratio);
    visitor.on_attribute("density", m_attrs.density);
    visitor.on_attribute("fixed_ratio", m_attrs.fixed_ratio);
    visitor.on_attribute("fixed_size", m_attrs.fixed_size);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("flip", m_attrs.flip);
    visitor.on_attribute("step", m_attrs.step);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variance);
    visitor.on_attribute("scale_all_sizes", m_attrs.scale_all_sizes);
    return true;
}

bool PriorBox::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_PriorBox_evaluate);
    return evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool op::v0::PriorBox::has_evaluate() const {
    OV_OP_SCOPE(v0_PriorBox_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

void PriorBox::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace v0
}  // namespace op
}  // namespace ov

// ------------------------------ V8 ------------------------------
namespace ov {
namespace op {
namespace v8 {
namespace {
template <element::Type_t ET>
bool evaluate(const Tensor& arg0, const Tensor& arg1, Tensor& out, const op::v8::PriorBox::Attributes& attrs) {
    using T = typename element_type_traits<ET>::value_type;
    ov::reference::prior_box(arg0.data<T>(), arg1.data<T>(), out.data<float>(), out.get_shape(), attrs);
    return true;
}

bool evaluate_prior_box(const Tensor& arg0,
                        const Tensor& arg1,
                        Tensor& out,
                        const op::v8::PriorBox::Attributes& attrs) {
    bool rc = true;
    switch (arg0.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
        OPENVINO_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

PriorBox::PriorBox(const Output<Node>& layer_shape, const Output<Node>& image_shape, const PriorBox::Attributes& attrs)
    : Op({layer_shape, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void PriorBox::validate_and_infer_types() {
    OV_OP_SCOPE(v8_PriorBox_validate_and_infer_types);

    const auto input_shapes = prior_box::validate::inputs_et(this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, element::f32, output_shapes.front());
    set_input_is_relevant_to_shape(0);
}

std::shared_ptr<Node> PriorBox::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_PriorBox_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PriorBox>(new_args.at(0), new_args.at(1), m_attrs);
}

int64_t PriorBox::number_of_priors(const PriorBox::Attributes& attrs) {
    // Starting with 0 number of prior and then various conditions on attributes will contribute
    // real number of prior boxes as PriorBox is a fat thing with several modes of
    // operation that will be checked in order in the next statements.
    int64_t num_priors = 0;

    // Total number of boxes around each point; depends on whether flipped boxes are included
    // plus one box 1x1.
    int64_t total_aspect_ratios = normalized_aspect_ratio(attrs.aspect_ratio, attrs.flip).size();

    if (attrs.scale_all_sizes)
        num_priors = total_aspect_ratios * attrs.min_size.size() + attrs.max_size.size();
    else
        num_priors = total_aspect_ratios + attrs.min_size.size() - 1;

    if (!attrs.fixed_size.empty())
        num_priors = total_aspect_ratios * attrs.fixed_size.size();

    for (auto density : attrs.density) {
        auto rounded_density = static_cast<int64_t>(density);
        auto density_2d = (rounded_density * rounded_density - 1);
        if (!attrs.fixed_ratio.empty())
            num_priors += attrs.fixed_ratio.size() * density_2d;
        else
            num_priors += total_aspect_ratios * density_2d;
    }
    return num_priors;
}

std::vector<float> PriorBox::normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip) {
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio) {
        unique_ratios.insert(std::round(ratio * 1e6f) / 1e6f);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6f) / 1e6f);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

bool PriorBox::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_PriorBox_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("max_size", m_attrs.max_size);
    visitor.on_attribute("aspect_ratio", m_attrs.aspect_ratio);
    visitor.on_attribute("density", m_attrs.density);
    visitor.on_attribute("fixed_ratio", m_attrs.fixed_ratio);
    visitor.on_attribute("fixed_size", m_attrs.fixed_size);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("flip", m_attrs.flip);
    visitor.on_attribute("step", m_attrs.step);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variance);
    visitor.on_attribute("scale_all_sizes", m_attrs.scale_all_sizes);
    visitor.on_attribute("min_max_aspect_ratios_order", m_attrs.min_max_aspect_ratios_order);
    return true;
}

bool PriorBox::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v8_PriorBox_evaluate);
    return evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool PriorBox::has_evaluate() const {
    OV_OP_SCOPE(v8_PriorBox_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        break;
    }
    return false;
}

void PriorBox::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
