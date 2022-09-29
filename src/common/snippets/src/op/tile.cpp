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

TileBase::TileBase(const std::vector<Output<Node>> &args, size_t dimension, size_t workAmount, size_t increment,
                   std::vector<bool> apply_increment, std::vector<int64_t> finalization_offsets)
        : Op(args), dimension(dimension), workAmount(workAmount), increment(increment),
          apply_increment(std::move(apply_increment)), finalization_offsets(std::move(finalization_offsets)) {
}

TileBase::TileBase(const std::vector<Output<Node>> &args, size_t dimension, size_t workAmount, size_t increment,
                   std::vector<bool> apply_increment)
        : Op(args), dimension(dimension), workAmount(workAmount), increment(increment),
        apply_increment(std::move(apply_increment)) {
    finalization_offsets = std::vector<int64_t>(args.size(), 0);
}

TileBase::TileBase(const std::vector<Output<Node>> &args, size_t dimension, size_t workAmount, size_t increment)
        : Op(args), dimension(dimension), workAmount(workAmount), increment(increment) {
    apply_increment = std::vector<bool> (args.size(), true);
    finalization_offsets = std::vector<int64_t>(args.size(), 0);
}

//std::shared_ptr<Node> TileBase::clone_with_new_inputs(const OutputVector& inputs) const {
//    return std::make_shared<TileBase>(inputs, tileRank, workAmount, increment);
//}

bool TileBase::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("dimension", dimension);
    visitor.on_attribute("work_amount", workAmount);
    visitor.on_attribute("increment", increment);
    return true;
}

TileBegin::TileBegin(const std::vector<Output<Node>> &args, size_t dimension, size_t workAmount, size_t increment,
                     std::vector<bool> apply_increments, std::vector<int64_t> finalization_offsets)
        : TileBase(args, dimension, workAmount, increment, std::move(apply_increments), std::move(finalization_offsets)),
        begin_address(nullptr) {
    constructor_validate_and_infer_types();
}
TileBegin::TileBegin(const std::vector<Output<Node>> &args, size_t dimension, size_t workAmount, size_t increment,
                     std::vector<bool> apply_increments)
        : TileBase(args, dimension, workAmount, increment, std::move(apply_increments)), begin_address(nullptr) {
    constructor_validate_and_infer_types();
}

TileBegin::TileBegin(const std::vector<Output<Node>> &args, size_t dimension, size_t workAmount, size_t increment)
    : TileBase(args, dimension, workAmount, increment), begin_address(nullptr) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> TileBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileBegin>(inputs, dimension, workAmount, increment);
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

TileEnd::TileEnd(const std::vector<Output<Node>> &args)
        : TileBase(args, 0, 0, 0) {
    const auto tileBegin = ov::as_type_ptr<TileBegin>(args.back().get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this, tileBegin != nullptr, "TileEnd must have TileBegin as the last argument");
    dimension = tileBegin->dimension;
    workAmount = tileBegin->workAmount;
    increment = tileBegin->increment;
    apply_increment = tileBegin->apply_increment;
    finalization_offsets = tileBegin->finalization_offsets;
    const auto io_size = get_output_size() + tileBegin->get_input_size();
    NODE_VALIDATION_CHECK(this, apply_increment.size() != io_size,
                          "TileEnd detected invalid size of the apply_increments arg: expected ",
                          io_size, " got ", apply_increment.size());
    NODE_VALIDATION_CHECK(this, finalization_offsets.size() != get_input_size(),
                          "TileEnd detected invalid size of the apply_increments arg: expected ",
                          io_size, " got ", apply_increment.size());
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> TileEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileEnd>(inputs);
}

void TileEnd::validate_and_infer_types() {
    const size_t num_inputs = get_input_size();
    const auto tileBegin = ov::as_type_ptr<TileBegin>(input(num_inputs - 1).get_source_output().get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this,
                          tileBegin != nullptr,
                          "The last argument of TileEnd must be TileBegin");
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

} // namespace op
} // namespace snippets
} // namespace ngraph