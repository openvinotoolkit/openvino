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

TileBegin::TileBegin(const std::vector<Output<Node>> &args, size_t tileRank)
    : Op(args), tileRank(tileRank) {
    constructor_validate_and_infer_types();
}

bool TileBegin::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("tileRank", tileRank);
    return true;
}

void TileBegin::validate_and_infer_types() {
    const size_t num_inputs = get_input_size();
    set_output_size(num_inputs + 1);
    const auto& ins = inputs();
    for (int i = 0; i < num_inputs; i++) {
        set_output_type(i, ins[i].get_element_type(), ins[i].get_partial_shape());
        // copy rt_info from inputs to outputs to pass reg_info for example
        const auto& rt_info_old = ins[i].get_source_output().get_tensor().get_rt_info();
        auto& rt_info_new = get_output_tensor(i).get_rt_info();
        rt_info_new = rt_info_old;
    }
    set_output_type(num_inputs, element::f32, ov::PartialShape{ov::Shape{}});
}

std::shared_ptr<Node> TileBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileBegin>(inputs, tileRank);
}

TileEnd::TileEnd(const std::vector<Output<Node>> &args, size_t tileRank)
    : Op(args), tileRank(tileRank) {
    constructor_validate_and_infer_types();
}

bool TileEnd::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("tileRank", tileRank);
    return true;
}

void TileEnd::validate_and_infer_types() {
    const size_t num_inputs = get_input_size();
    const auto tileBegin = ov::as_type_ptr<TileBegin>(input(num_inputs - 1).get_source_output().get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this,
                          tileBegin != nullptr,
                          "The last argument of TileEnd must be TileBegin");
    NODE_VALIDATION_CHECK(this, tileBegin->tileRank == tileRank,
                          "TileBegin and TileEnd have different tileRanks:", tileBegin->tileRank, " vs ", tileRank);
    set_output_size(num_inputs - 1);
    const auto& ins = inputs();
    for (int i = 0; i < num_inputs - 1; i++) {
        set_output_type(i, ins[i].get_element_type(), ins[i].get_partial_shape());
        // copy rt_info from inputs to outputs to pass reg_info for example
        const auto& rt_info_old = ins[i].get_source_output().get_tensor().get_rt_info();
        auto& rt_info_new = get_output_tensor(i).get_rt_info();
        rt_info_new = rt_info_old;
    }
}

std::shared_ptr<Node> TileEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileEnd>(inputs, tileRank);
}

} // namespace op
} // namespace snippets
} // namespace ngraph