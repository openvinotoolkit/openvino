// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include "bound_evaluate.hpp"
#include "concat_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/dimension.hpp"
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

    const auto output_shape = shape_infer(this, input_shapes).front();
    if (output_shape.rank().is_static() && (get_concatenation_axis() < 0)) {
        set_concatenation_axis(ov::util::normalize(get_axis(), output_shape.size()));
    }

    set_output_type(0, inputs_et, output_shape);
}

std::shared_ptr<Node> Concat::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Concat_clone_with_new_inputs);
    return std::make_shared<Concat>(new_args, m_axis);
}

template <typename T>
void evaluate_concat(const Concat* node, TensorVector& outputs, const TensorVector& inputs) {
    const auto inputs_count = inputs.size();
    std::vector<Shape> arg_shapes;
    std::vector<PartialShape> input_shapes;
    arg_shapes.reserve(inputs_count);
    input_shapes.reserve(inputs_count);

    std::vector<const T*> arg_bufs(inputs_count);
    auto arg_buf = arg_bufs.begin();
    for (auto& input : inputs) {
        *arg_buf = static_cast<const T*>(input.data());
        ++arg_buf;
        const auto& input_shape = input.get_shape();
        arg_shapes.emplace_back(input_shape);
        input_shapes.emplace_back(input_shape);
    }

    const auto& out_shape = shape_infer(node, input_shapes).front().to_shape();
    outputs.front().set_shape(out_shape);
    reference::concat(arg_bufs,
                      static_cast<T*>(outputs.front().data()),
                      arg_shapes,
                      out_shape,
                      ov::util::normalize(node->get_axis(), out_shape.size()),
                      outputs.front().get_element_type().size());
}

bool Concat::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Concat_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    if (outputs.front().get_element_type() == ov::element::string) {
        evaluate_concat<std::string>(this, outputs, inputs);
    } else {
        evaluate_concat<char>(this, outputs, inputs);
    }

    return true;
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
