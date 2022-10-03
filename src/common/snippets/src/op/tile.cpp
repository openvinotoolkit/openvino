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

TileBase::TileBase(const std::vector<Output<Node>> &args, size_t dimension, size_t work_amount, size_t increment)
        : Op(args), dimension(dimension), work_amount(work_amount), increment(increment) {
}

bool TileBase::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("dimension", dimension);
    visitor.on_attribute("work_amount", work_amount);
    visitor.on_attribute("increment", increment);
    return true;
}

size_t TileBase::get_work_amount() const {
    return work_amount;
}

size_t TileBase::get_increment() const {
    return increment;
}

size_t TileBase::get_dimension() const {
    return dimension;
}

TileBegin::TileBegin(const std::vector<Output<Node>> &args, size_t dimension, size_t work_amount, size_t increment)
        : TileBase(args, dimension, work_amount, increment),
        begin_address(nullptr), input_regs({}) {
    // We can only call a reduced validate_and_infer types from the constructor, since TileEnd might not be attached
    // to the TileBegin at this point (which is usually the case: create TileBegin first => then attach TileEnd to it)
    validate_and_infer_types_except_TileEnd();
}

TileBegin::TileBegin(const std::vector<Output<Node>> &args)
        : TileBase(args, 0, 0, 0), begin_address(nullptr), input_regs({}) {
    validate_and_infer_types_except_TileEnd();
}

std::shared_ptr<Node> TileBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::shared_ptr<TileBegin>(new TileBegin(inputs, dimension, work_amount, increment));
}


void TileBegin::validate_and_infer_types_except_TileEnd() {
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

void TileBegin::validate_and_infer_types() {
    validate_and_infer_types_except_TileEnd();
    const auto& last_output_inputs = output(get_output_size() - 1).get_target_inputs();
    NODE_VALIDATION_CHECK(this, last_output_inputs.size() == 1, "TileBegin must have exactly one input attached to the last output");
    const auto& tile_end = ov::as_type_ptr<TileEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    NODE_VALIDATION_CHECK(this, tile_end != nullptr, "TileBegin must have TileEnd connected to its last output");
    dimension = tile_end->get_dimension();
    work_amount = tile_end->get_work_amount();
    increment = tile_end->get_increment();
}

std::shared_ptr<TileEnd> TileBegin::get_tile_end() {
    const auto& last_output_inputs = output(get_output_size() - 1).get_target_inputs();
    if (last_output_inputs.size() != 1)
        throw std::invalid_argument("TileBegin has more than one inputs attached to the last output");
    const auto& tile_end = ov::as_type_ptr<TileEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    if (!tile_end)
        throw std::invalid_argument("TileBegin last output is not connected to TileEnd");
    return  tile_end;
}

TileEnd::TileEnd(const std::vector<Output<Node>> &args, size_t dimension, size_t work_amount, size_t increment,
                 std::vector<bool> apply_increment, std::vector<int64_t> finalization_offsets)
        : TileBase(args, dimension, work_amount, increment), apply_increment(std::move(apply_increment)),
        finalization_offsets(std::move(finalization_offsets)) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> TileEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<TileEnd>(inputs, dimension, work_amount, increment, apply_increment, finalization_offsets);
}

std::shared_ptr<TileBegin> TileEnd::get_tile_begin() {
    const auto& tile_begin = ov::as_type_ptr<TileBegin>(get_input_source_output(get_input_size() - 1).get_node_shared_ptr());
    if (!tile_begin)
        throw std::invalid_argument("TileEnd last input is not connected to TileBegin");
    return  tile_begin;
}

const std::vector<int64_t>& TileEnd::get_finalization_offsets() const {
    return finalization_offsets;
}

const std::vector<bool>& TileEnd::get_apply_increment() const {
    return apply_increment;
}

void TileEnd::set_finalization_offsets(std::vector<int64_t> offsets) {
    if (offsets.size() != tile_io_size)
        throw std::invalid_argument("TileEnd set_finalization_offsets is called with inconsistent offsets.size()");
    finalization_offsets = std::move(offsets);
}

void TileEnd::set_apply_increment(std::vector<bool> allow_increment) {
    if (allow_increment.size() != tile_io_size)
        throw std::invalid_argument("TileEnd set_apply_increment is called with inconsistent apply_increment.size()");
    apply_increment = std::move(allow_increment);
}

void TileEnd::set_work_amount(size_t new_work_amount) {
    work_amount = new_work_amount;
    // Update TileBegin to maintain consistency between the Tiles
    get_tile_begin()->work_amount = new_work_amount;
}

void TileEnd::set_increment(size_t new_increment) {
    increment = new_increment;
    // Update TileBegin to maintain consistency between the Tiles
    get_tile_begin()->increment = new_increment;
}

void TileEnd::validate_and_infer_types() {
    const size_t num_inputs = get_input_size();
    const auto tileBegin = ov::as_type_ptr<TileBegin>(input(get_input_size() - 1).get_source_output().get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this, tileBegin != nullptr, "TileEnd must have TileBegin as the last argument");
    // Note: have to -2 because the TileBegin->TileEnd edge is counted twice
    tile_io_size = get_input_size() + tileBegin->get_output_size() - 2;
    NODE_VALIDATION_CHECK(this, apply_increment.empty() || apply_increment.size() == tile_io_size,
                          "apply_increments must be either empty or defined per every input & output of joined Tile. Expected size: ",
                          tile_io_size, " got ", apply_increment.size());
    NODE_VALIDATION_CHECK(this, finalization_offsets.empty() || finalization_offsets.size() == tile_io_size,
                          "finalization_offsets must be either empty or defined per every input & output of joined Tile. Expected size: ",
                          tile_io_size, " got ", finalization_offsets.size());
    if (apply_increment.empty())
        apply_increment.resize(tile_io_size, true);
    if (finalization_offsets.empty())
        finalization_offsets.resize(tile_io_size, 0);
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