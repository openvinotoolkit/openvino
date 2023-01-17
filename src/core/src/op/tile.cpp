// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/tile.hpp"

#include <ngraph/validation_util.hpp>
#include <tile_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/tile.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ngraph;

op::v0::Tile::Tile(const Output<Node>& data, const Output<Node>& repeats) : Op({data, repeats}) {
    ov::mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Tile::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Tile_visit_attributes);
    return true;
}

void op::v0::Tile::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Tile_validate_and_infer_types);

    // Repeats should have integer data type. For now we only allow i64
    const auto& repeats_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          repeats_et.is_integral(),
                          "Tile repeats must have any integer element type, but has ",
                          repeats_et);

    const auto input_shapes = get_node_input_partial_shapes(*this);
    auto output_shapes = std::vector<PartialShape>(1, ov::PartialShape{});
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);

    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::v0::Tile::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Tile_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Tile>(new_args.at(0), new_args.at(1));
}

bool op::v0::Tile::evaluate_tile(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    const auto& data = inputs[0];
    const auto& axis = inputs[1];
    auto& output = outputs[0];
    auto repeats_val = read_index_vector(axis);
    const auto repeats_rank = repeats_val.size();

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {data->get_shape(), axis->get_shape()};
    shape_infer(this, input_shapes, output_shapes, {{1, axis}});
    const auto& output_shape = output_shapes[0].to_shape();
    if (!output->get_is_allocated()) {
        output->set_shape(output_shape);
    }
    repeats_val.insert(repeats_val.begin(), output_shape.size() - repeats_rank, 1);
    ngraph::runtime::reference::tile(data->get_data_ptr<const char>(),
                                     output->get_data_ptr<char>(),
                                     data->get_shape(),
                                     output_shape,
                                     data->get_element_type().size(),
                                     repeats_val);

    return true;
}

bool op::v0::Tile::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Tile_evaluate);
    return evaluate_tile(outputs, inputs);
}

bool op::v0::Tile::has_evaluate() const {
    OV_OP_SCOPE(v0_Tile_has_evaluate);
    return true;
}

bool op::v0::Tile::evaluate_lower(const HostTensorVector& output_values) const {
    OV_OP_SCOPE(v0_Tile_evaluate_lower);

    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool op::v0::Tile::evaluate_upper(const HostTensorVector& output_values) const {
    OV_OP_SCOPE(v0_Tile_evaluate_upper);

    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool op::v0::Tile::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(v0_Tile_evaluate_label);
    OPENVINO_ASSERT(output_labels.size() == 1);

    return get_input_tensor(1).has_and_set_bound() && default_label_evaluator(this, output_labels);
}
