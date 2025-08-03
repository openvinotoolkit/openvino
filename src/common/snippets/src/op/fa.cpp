// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/fa.hpp"

#include <cstddef>
#include <iterator>
#include <memory>
#include <set>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::op {

namespace {
std::vector<size_t> get_output_layout(const std::shared_ptr<const ov::Node>& n) {
    const auto& key = lowered::PortDescriptorVectorAttribute::get_type_info_static();
    const auto& rt_info = n->get_rt_info();
    const auto& found = rt_info.find(key);
    if (found != rt_info.end()) {
        const auto& out_descs = found->second.as<lowered::PortDescriptorVectorAttribute>().outputs;
        if (out_descs.size() != n->get_output_size()) {
            OPENVINO_THROW("Get output port descriptor is failed: incorrect count");
        }
        const auto& port_desc = out_descs[0];
        return port_desc->get_layout();
    }
    return {};
}

}  // namespace

FA::FA(const Output<Node>& q, const Output<Node>& k, const Output<Node>& v)
    : MemoryAccess(std::set<size_t>{0, 1, 2}, std::set<size_t>{0}),
      Op({q, k, v}) {
    set_output_size(1);
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> FA::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(FA_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<FA>(new_args.at(0), new_args.at(1), new_args.at(2));
}

std::vector<ov::PartialShape> FA::get_planar_input_shapes(const std::vector<ov::Input<ov::Node>>& inputs) {
    // assert(inputs.size() == 3 && "FA::get_planar_input_shapes() expects 3 inputs");
    return {utils::get_planar_pshape(inputs[0]),
            utils::get_planar_pshape(inputs[1]),
            utils::get_planar_pshape(inputs[2])};
}

ov::PartialShape FA::get_planar_output_shape(const ov::PartialShape& output_shape) const {
    // This method can be safely called from validate_and_infer_types() before output creation
    const auto& out_layout = get_output_layout(shared_from_this());
    if (!out_layout.empty()) {
        return utils::get_planar_pshape(output_shape, {});
    }

    return output_shape;
}

void FA::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(FA_validate_and_infer_types);

    const auto planar_input_shapes = get_planar_input_shapes(inputs());
    auto output_shape = infer_output_partial_shape(planar_input_shapes);
    const auto in_type = get_input_element_type(0);
    set_output_type(0, in_type, get_planar_output_shape(output_shape));
}

ov::PartialShape FA::infer_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 3, "FA expects 3 input shapes for shape inference");

    const auto& arg_q_shape = input_shapes[0];
    const auto& arg_v_shape = input_shapes[2];
    size_t arg_q_rank = arg_q_shape.size(), arg_v_rank = arg_v_shape.size();

    // temporary shapes to calculate output shape
    ov::PartialShape arg_q_shape_tmp(arg_q_shape), arg_v_shape_tmp(arg_v_shape);

    // one-dimensional tensors unsqueezing is applied to each input independently.
    if (arg_q_rank == 1) {
        // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
        // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
        // For example {S} will be reshaped to {1, S}.
        arg_q_shape_tmp.insert(arg_q_shape_tmp.begin(), 1);
        arg_q_rank = arg_q_shape_tmp.size();
    }
    if (arg_v_rank == 1) {
        // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
        // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
        // For example {S} will be reshaped to {S, 1}.
        arg_v_shape_tmp.insert(arg_v_shape_tmp.end(), 1);
        arg_v_rank = arg_v_shape_tmp.size();
    }

    // add 1 to begin to align shape ranks if needed
    if (arg_q_rank < arg_v_rank) {
        arg_q_shape_tmp.insert(arg_q_shape_tmp.begin(), arg_v_rank - arg_q_rank, 1);
    } else if (arg_q_rank > arg_v_rank) {
        arg_v_shape_tmp.insert(arg_v_shape_tmp.begin(), arg_q_rank - arg_v_rank, 1);
    }

    using DimType = typename std::iterator_traits<typename ov::PartialShape::iterator>::value_type;
    size_t max_rank = arg_q_shape_tmp.size();
    std::vector<DimType> output_shape(max_rank);
    for (size_t i = 0; i < max_rank - 2; ++i) {
        OPENVINO_ASSERT(DimType::broadcast_merge(output_shape[i], arg_q_shape_tmp[i], arg_v_shape_tmp[i]) ||
                            arg_q_shape_tmp[i].is_dynamic() || arg_v_shape_tmp[i].is_dynamic(),
                        "Incompatible fa batch dimension");
    }
    output_shape[output_shape.size() - 2] = arg_q_shape_tmp[arg_q_shape_tmp.size() - 2];  // M in Q
    output_shape[output_shape.size() - 1] = arg_v_shape_tmp[arg_v_shape_tmp.size() - 1];  // N in V

    // removing the temporary axes from originally 1D tensors.
    if (arg_q_shape.size() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 2);
    }
    if (arg_v_shape.size() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 1);
    }
    return output_shape;
}

}  // namespace ov::snippets::op
