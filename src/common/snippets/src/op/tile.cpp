// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/tile.hpp"
#include "snippets/generator.hpp"

using namespace std;
namespace ngraph {
namespace snippets {
namespace op {

Tile::Tile(const std::vector<AllocatedEmitter> &region, size_t increment,
                         size_t num_inputs, size_t num_outputs,
                         const std::vector<size_t> &io_dims, const std::vector<size_t> &io_data_sizes) :
        Op(), region(region), increment(increment), num_inputs(num_inputs), num_outputs(num_outputs), io_dims(io_dims),
        io_data_size(io_data_sizes) {
}

TileBegin::TileBegin(const std::vector<Output<Node>> &args) : Op(args) {
    constructor_validate_and_infer_types();
}

bool TileBegin::visit_attributes(AttributeVisitor &visitor) {
    //todo add significant fields to attribute visitor
    return true;
}

void TileBegin::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> TileBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileBegin>(inputs);
}

TileEnd::TileEnd(const std::vector<Output<Node>> &args) : Op(args) {
    constructor_validate_and_infer_types();
}

bool TileEnd::visit_attributes(AttributeVisitor &visitor) {
    //todo add significant fields to attribute visitor
    return true;
}

void TileEnd::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> TileEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileEnd>(inputs);
}

} // namespace op
} // namespace snippets
} // namespace ngraph