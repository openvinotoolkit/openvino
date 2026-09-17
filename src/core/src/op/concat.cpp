// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include <algorithm>
#include <type_traits>

#include "bound_evaluate.hpp"
#include "concat_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/concat.hpp"

namespace ov {
namespace op {
namespace v0 {
namespace {
template <class T>
std::vector<const T*> get_data_ptrs(const TensorVector& inputs) {
    std::vector<const T*> ptrs;
    ptrs.reserve(inputs.size());
    for (auto& input : inputs) {
        if constexpr (std::is_same_v<T, std::string>) {
            ptrs.emplace_back(input.data<T>());
        } else {
            ptrs.emplace_back(static_cast<const T*>(input.data()));
        }
    }
    return ptrs;
}

enum class concat_kind { unsupported, string, packed, regular };

concat_kind get_concat_kind(const element::Type& elem_type,
                            const std::vector<Shape>& input_shapes,
                            const Shape& output_shape,
                            size_t axis) {
    if (elem_type == element::string) {
        return concat_kind::string;
    } else if (const auto bitwidth = elem_type.bitwidth(); bitwidth >= 8) {
        return concat_kind::regular;
    } else if (bitwidth == 0 || elem_type == element::u3 || elem_type == element::u6) {
        return concat_kind::unsupported;
    } else {
        const auto steps = ov::shape_size(output_shape.begin(), output_shape.begin() + axis);
        const auto is_misaligned = [steps, bitwidth](auto&& shape) {
            return (ov::shape_size(shape) / steps * bitwidth) % 8 != 0;
        };
        return steps == 0 || std::none_of(input_shapes.begin(), input_shapes.end(), is_misaligned)
                   ? concat_kind::packed
                   : concat_kind::unsupported;
    }
}
}  // namespace

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
    arg_shapes.reserve(inputs_count);
    input_shapes.reserve(inputs_count);

    for (auto& input : inputs) {
        const auto& input_shape = input.get_shape();
        arg_shapes.emplace_back(input_shape);
        input_shapes.emplace_back(input_shape);
    }

    const auto& elem_type = outputs[0].get_element_type();
    const auto& out_shape = shape_infer(this, input_shapes).front().to_shape();
    const auto axis = ov::util::normalize(get_axis(), out_shape.size());

    switch (get_concat_kind(elem_type, arg_shapes, out_shape, axis)) {
    case concat_kind::string:
        outputs.front().set_shape(out_shape);
        reference::concat(get_data_ptrs<std::string>(inputs),
                          outputs[0].data<std::string>(),
                          arg_shapes,
                          out_shape,
                          axis);
        return true;
    case concat_kind::packed:
        outputs.front().set_shape(out_shape);
        reference::concat(get_data_ptrs<int8_t>(inputs),
                          outputs[0].data<int8_t>(),
                          arg_shapes,
                          out_shape,
                          axis,
                          elem_type.bitwidth());
        return true;
    case concat_kind::regular:
        outputs.front().set_shape(out_shape);
        reference::concat(get_data_ptrs<char>(inputs),
                          static_cast<char*>(outputs[0].data()),
                          arg_shapes,
                          out_shape,
                          axis,
                          elem_type.size());
        return true;
    case concat_kind::unsupported:
    default:
        return false;
    }
}

bool Concat::has_evaluate() const {
    OV_OP_SCOPE(v0_Concat_has_evaluate);
    return true;
}

bool Concat::evaluate_lower(TensorVector& output_values) const {
    return default_lower_bound_evaluator(this, output_values);
}

bool Concat::evaluate_upper(TensorVector& output_values) const {
    return default_upper_bound_evaluator(this, output_values);
}

bool Concat::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return default_symbol_evaluator(this, {}, output_symbols);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
