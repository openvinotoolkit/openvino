// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include "bound_evaluate.hpp"
#include "concat_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/concat.hpp"

namespace ov {
namespace op {
namespace v0 {

Concat::Concat(const OutputVector& args, int64_t axis) : Op(args), m_axis(axis) {
    constructor_validate_and_infer_types();
}

Concat::Concat(const NodeVector& args, int64_t axis) : Concat(as_output_vector(args), axis) {}

bool Concat::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Concat_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void Concat::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Concat_validate_and_infer_types);
    element::Type inputs_et{element::dynamic};
    auto input_shapes = std::vector<PartialShape>();

    for (size_t i = 0; i < get_input_size(); ++i) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
        input_shapes.push_back(get_input_partial_shape(i));
    }

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, inputs_et, output_shapes[0]);
}

std::shared_ptr<Node> Concat::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Concat_clone_with_new_inputs);
    return std::make_shared<Concat>(new_args, m_axis);
}

bool Concat::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Concat_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto inputs_count = inputs.size();
    std::vector<Shape> arg_shapes;
    std::vector<PartialShape> input_shapes;
    std::vector<const char*> arg_bufs;
    arg_shapes.reserve(inputs_count);
    input_shapes.reserve(inputs_count);
    arg_bufs.reserve(inputs_count);

    for (auto& input : inputs) {
        const auto& input_shape = input.get_shape();
        arg_shapes.emplace_back(input_shape);
        input_shapes.emplace_back(input_shape);
        arg_bufs.emplace_back(static_cast<const char*>(input.data()));
    }

    const auto& out_shape = shape_infer(this, input_shapes).front().to_shape();
    outputs.front().set_shape(out_shape);
    const auto elem_type = outputs.front().get_element_type();
    reference::concat(arg_bufs,
                      static_cast<char*>(outputs.front().data()),
                      arg_shapes,
                      out_shape,
                      ov::util::normalize(this->get_axis(), out_shape.size()),
                      elem_type.size(),
                      elem_type);

    return true;
}

bool Concat::has_evaluate() const {
    OV_OP_SCOPE(v0_Concat_has_evaluate);
    return true;
}

namespace helpers {
template <class T>
Tensor make_tensor(const element::Type_t et, const Shape& shape, T val) {
    Tensor t{et, shape};
    for (size_t i = 0; i < t.get_size(); ++i) {
        t.data<T>()[i] = val;
    }
    return t;
}

static Tensor make_tensor_of_min_value(const element::Type_t et, const Shape& shape) {
#define CASE(type)                                                                     \
    case element::type: {                                                              \
        typename ov::fundamental_type_for<element::type> value = 0;                    \
        return make_tensor<ov::fundamental_type_for<element::type>>(et, shape, value); \
    }

    switch (et) {
        CASE(boolean);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(f64);
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        CASE(u1);
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
    default:
        return {};
    }
#undef CASE
}

static Tensor make_tensor_of_max_value(const element::Type_t et, const Shape& shape) {
#define CASE(type)                                                                     \
    case element::type: {                                                              \
        using T = typename ov::fundamental_type_for<element::type>;                    \
        const T value = std::numeric_limits<T>::max();                                 \
        return make_tensor<ov::fundamental_type_for<element::type>>(et, shape, value); \
    }

    switch (et) {
        CASE(boolean);
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(f64);
        CASE(i8);
        CASE(i16);
        CASE(i32);
        CASE(i64);
        CASE(u1);
        CASE(u8);
        CASE(u16);
        CASE(u32);
        CASE(u64);
    default:
        return {};
    }
#undef CASE
}

static bool evaluate_bound(const Node* node,
                           const ov::Tensor& (ov::descriptor::Tensor::*get_bound)() const,
                           Tensor (*make_tensor_func)(const element::Type_t, const Shape&),
                           ov::TensorVector& output_values) {
    const auto size = node->get_input_size();
    ov::TensorVector inputs;
    inputs.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        ov::descriptor::Tensor& ts = node->get_input_tensor(i);
        if (auto bound = (ts.*get_bound)()) {
            inputs.push_back(bound);
        } else {
            inputs.push_back(make_tensor_func(ts.get_element_type(), ts.get_shape()));
        }
    }
    return node->evaluate(output_values, inputs);
}

}  // namespace helpers

bool Concat::evaluate_lower(TensorVector& output_values) const {
    return helpers::evaluate_bound(this,
                                   &ov::descriptor::Tensor::get_lower_value,
                                   helpers::make_tensor_of_min_value,
                                   output_values);
}

bool Concat::evaluate_upper(TensorVector& output_values) const {
    return helpers::evaluate_bound(this,
                                   &ov::descriptor::Tensor::get_upper_value,
                                   helpers::make_tensor_of_max_value,
                                   output_values);
}

bool Concat::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return default_symbol_evaluator(this, {}, output_symbols);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
