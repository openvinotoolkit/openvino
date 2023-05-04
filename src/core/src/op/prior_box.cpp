// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/prior_box.hpp"

#include <array>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/prior_box.hpp"
#include "openvino/runtime/tensor.hpp"
#include "prior_box_shape_inference.hpp"

using namespace std;
using namespace ngraph;

namespace ov {
namespace op {
namespace prior_box {
static constexpr auto input_names = std::array<const char*, 2>{"output size", "image"};

namespace validate {
std::vector<PartialShape> inputs_et(const Node* const op) {
    const auto inputs_size = op->get_input_size();
    auto input_shapes = std::vector<PartialShape>();
    input_shapes.reserve(inputs_size);

    for (size_t i = 0; i < inputs_size; ++i) {
        const auto& et = op->get_input_element_type(i);
        NODE_VALIDATION_CHECK(op,
                              et.is_integral_number(),
                              prior_box::input_names[i],
                              " input must be an integral number, but is: ",
                              et);
        input_shapes.push_back(op->get_input_partial_shape(i));
    }
    return input_shapes;
}
}  // namespace validate
}  // namespace prior_box
}  // namespace op
}  // namespace ov

op::v0::PriorBox::PriorBox(const Output<Node>& layer_shape,
                           const Output<Node>& image_shape,
                           const PriorBox::Attributes& attrs)
    : Op({layer_shape, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::v0::PriorBox::validate_and_infer_types() {
    OV_OP_SCOPE(v0_PriorBox_validate_and_infer_types);

    const auto input_shapes = prior_box::validate::inputs_et(this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, element::f32, output_shapes.front());
    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v0::PriorBox::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PriorBox_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<PriorBox>(new_args.at(0), new_args.at(1), m_attrs);
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

std::vector<float> op::v0::PriorBox::normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip) {
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio) {
        unique_ratios.insert(std::round(ratio * 1e6f) / 1e6f);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6f) / 1e6f);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

bool op::v0::PriorBox::visit_attributes(AttributeVisitor& visitor) {
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

namespace ov {
namespace op {
namespace prior_box {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::v0::PriorBox::Attributes& attrs) {
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
    ngraph::runtime::reference::prior_box(arg0->get_data_ptr<ET>(),
                                          arg1->get_data_ptr<ET>(),
                                          out->get_data_ptr<float>(),
                                          out->get_shape(),
                                          attrs_v8);
    return true;
}

bool evaluate_prior_box(const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out,
                        const op::v0::PriorBox::Attributes& attrs) {
    bool rc = true;
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace prior_box

namespace v0 {

void PriorBox::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace v0

namespace v8 {
void PriorBox::set_attrs(Attributes attrs) {
    m_attrs = std::move(attrs);
}
}  // namespace v8
}  // namespace op
}  // namespace ov

bool op::v0::PriorBox::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_PriorBox_evaluate);
    return prior_box::evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool op::v0::PriorBox::has_evaluate() const {
    OV_OP_SCOPE(v0_PriorBox_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}

// ------------------------------ V8 ------------------------------
op::v8::PriorBox::PriorBox(const Output<Node>& layer_shape,
                           const Output<Node>& image_shape,
                           const PriorBox::Attributes& attrs)
    : Op({layer_shape, image_shape}),
      m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::v8::PriorBox::validate_and_infer_types() {
    OV_OP_SCOPE(v8_PriorBox_validate_and_infer_types);

    const auto input_shapes = prior_box::validate::inputs_et(this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, element::f32, output_shapes.front());
    set_input_is_relevant_to_shape(0);
}

shared_ptr<Node> op::v8::PriorBox::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_PriorBox_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<PriorBox>(new_args.at(0), new_args.at(1), m_attrs);
}

int64_t op::v8::PriorBox::number_of_priors(const PriorBox::Attributes& attrs) {
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

std::vector<float> op::v8::PriorBox::normalized_aspect_ratio(const std::vector<float>& aspect_ratio, bool flip) {
    std::set<float> unique_ratios;
    for (auto ratio : aspect_ratio) {
        unique_ratios.insert(std::round(ratio * 1e6f) / 1e6f);
        if (flip)
            unique_ratios.insert(std::round(1 / ratio * 1e6f) / 1e6f);
    }
    unique_ratios.insert(1);
    return std::vector<float>(unique_ratios.begin(), unique_ratios.end());
}

bool op::v8::PriorBox::visit_attributes(AttributeVisitor& visitor) {
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

namespace prior_box_v8 {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::v8::PriorBox::Attributes& attrs) {
    runtime::reference::prior_box(arg0->get_data_ptr<ET>(),
                                  arg1->get_data_ptr<ET>(),
                                  out->get_data_ptr<float>(),
                                  out->get_shape(),
                                  attrs);
    return true;
}

bool evaluate_prior_box(const HostTensorPtr& arg0,
                        const HostTensorPtr& arg1,
                        const HostTensorPtr& out,
                        const op::v8::PriorBox::Attributes& attrs) {
    bool rc = true;
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_prior_box, i8, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i16, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i32, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, i64, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u8, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u16, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u32, arg0, arg1, out, attrs);
        NGRAPH_TYPE_CASE(evaluate_prior_box, u64, arg0, arg1, out, attrs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace prior_box_v8

bool op::v8::PriorBox::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v8_PriorBox_evaluate);
    return prior_box_v8::evaluate_prior_box(inputs[0], inputs[1], outputs[0], get_attrs());
}

bool op::v8::PriorBox::has_evaluate() const {
    OV_OP_SCOPE(v8_PriorBox_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        return true;
    default:
        break;
    }
    return false;
}
