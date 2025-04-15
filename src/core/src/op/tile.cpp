// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tile.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/tile.hpp"
#include "tile_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {

Tile::Tile(const Output<Node>& data, const Output<Node>& repeats) : Op({data, repeats}) {
    ov::mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

void Tile::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Tile_validate_and_infer_types);

    // Repeats should have integer data type. For now we only allow i64
    const auto& repeats_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          repeats_et.is_integral(),
                          "Tile repeats must have any integer element type, but has ",
                          repeats_et);
    auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
}

std::shared_ptr<Node> Tile::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Tile_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Tile>(new_args.at(0), new_args.at(1));
}

bool Tile::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Tile_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    const auto& d = inputs[0];
    const auto& r = inputs[1];
    auto repeats = get_tensor_data_as<int64_t>(r);

    const std::vector<ov::PartialShape> input_shapes{d.get_shape(), r.get_shape()};
    const auto output_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();
    outputs[0].set_shape(output_shape);
    repeats.insert(repeats.begin(), output_shape.size() - repeats.size(), 1);
    reference::tile(static_cast<const char*>(d.data()),
                    static_cast<char*>(outputs[0].data()),
                    d.get_shape(),
                    output_shape,
                    d.get_element_type().size(),
                    repeats);

    return true;
}

bool Tile::has_evaluate() const {
    OV_OP_SCOPE(v0_Tile_has_evaluate);
    return true;
}

bool Tile::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v0_Tile_evaluate_lower);

    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Tile::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v0_Tile_evaluate_upper);

    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Tile::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    OV_OP_SCOPE(v0_Tile_evaluate_symbol);
    OPENVINO_ASSERT(output_symbols.size() == 1);

    return get_input_tensor(1).has_and_set_bound() && ov::util::default_symbol_evaluator(this, output_symbols);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
