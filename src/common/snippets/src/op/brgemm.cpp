// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"
#include "snippets/op/brgemm.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "snippets/utils.hpp"

namespace ngraph {
namespace snippets {
namespace op {

Brgemm::Brgemm(const Output<Node>& A, const Output<Node>& B,
               const size_t offset_a, const size_t offset_b, const size_t offset_c) : MemoryAccess({A, B}, 2, 1) {
    set_output_size(1);
    set_input_offset(offset_a, 0);
    set_input_offset(offset_b, 1);
    set_output_offset(offset_a, 0);
    constructor_validate_and_infer_types();
}

void Brgemm::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Brgemm_validate_and_infer_types);
    // If no leading dimensions are provided, assume dense row-major inputs-outputs
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static(),
                          "Brgemm currently supports only static shapes.");

    std::vector<ov::PartialShape> planar_input_shapes = {
            utils::get_port_planar_shape(input_value(0)),
            utils::get_port_planar_shape(input_value(1))
    };

    auto output_shape = get_output_partial_shape(planar_input_shapes);
    const auto& output_layout = utils::get_node_output_layout(this);
    set_output_type(0,
                    get_output_type(),
                    utils::get_reordered_planar_shape(output_shape, output_layout));
}

std::shared_ptr<Node> Brgemm::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Brgemm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Brgemm>(new_args.at(0), new_args.at(1), get_offset_a(), get_offset_b(), get_offset_c());
}

ov::element::Type Brgemm::get_output_type() const {
    const auto element_type_a = get_input_element_type(0);
    const auto element_type_b = get_input_element_type(1);
    const bool is_f32 = utils::everyone_is(element::f32, element_type_a, element_type_b);
    const bool is_int8 = utils::one_of(element_type_a, element::i8, element::u8) && element_type_b == element::i8;
    const bool is_bf16 = utils::everyone_is(element::bf16, element_type_a, element_type_b);
    if (is_f32 || is_bf16) {
       return element::f32;
    } else if (is_int8) {
        return element::i32;
    } else {
        OPENVINO_THROW("BrgemmCPU node has incompatible input element types: " +
                            element_type_a.get_type_name() +
                            " and " +
                            element_type_b.get_type_name());
    }
}

ov::PartialShape Brgemm::get_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) const {
    NGRAPH_CHECK(input_shapes.size() == 2, "BRGEMM expects 2 input shapes for shape inference");

    // Note: All majors checks are missed because Brgemm is transformed from MatMul with whole shape infer support

    const auto arg0_shape = input_shapes[0];
    const auto arg1_shape = input_shapes[1];

    size_t arg0_rank = arg0_shape.size(), arg1_rank = arg1_shape.size();

    // temporary shapes to calculate output shape
    ov::PartialShape arg0_shape_tmp(arg0_shape), arg1_shape_tmp(arg1_shape);

    // one-dimensional tensors unsqueezing is applied to each input independently.
    if (arg0_rank == 1) {
        // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
        // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
        // For example {S} will be reshaped to {1, S}.
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), 1);
        arg0_rank = arg0_shape_tmp.size();
    }
    if (arg1_rank == 1) {
        // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
        // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
        // For example {S} will be reshaped to {S, 1}.
        arg1_shape_tmp.insert(arg1_shape_tmp.end(), 1);
        arg1_rank = arg1_shape_tmp.size();
    }
    // Check matrices dimensions compatibility,
    using DimType = typename std::iterator_traits<typename ov::PartialShape::iterator>::value_type;
    auto merged_dimension = DimType();
    auto arg0_col_dim = arg0_shape_tmp[arg0_rank - 1];
    auto arg1_row_dim = arg1_shape_tmp[arg1_rank - 2];
    OPENVINO_ASSERT(DimType::merge(merged_dimension, arg0_col_dim, arg1_row_dim) || arg0_col_dim.is_dynamic() || arg1_row_dim.is_dynamic(),
                    "Incompatible Brgemm matrix dimension");

    // add 1 to begin to align shape ranks if needed
    if (arg0_rank < arg1_rank)
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), arg1_rank - arg0_rank, 1);
    else if (arg0_rank > arg1_rank)
        arg1_shape_tmp.insert(arg1_shape_tmp.begin(), arg0_rank - arg1_rank, 1);

    size_t max_rank = arg0_shape_tmp.size();
    std::vector<DimType> output_shape(max_rank);
    for (size_t i = 0; i < max_rank - 2; ++i) {
         OPENVINO_ASSERT(DimType::broadcast_merge(output_shape[i], arg0_shape_tmp[i], arg1_shape_tmp[i]) ||
                         arg0_shape_tmp[i].is_dynamic() ||
                         arg1_shape_tmp[i].is_dynamic(),
                        "Incompatible Brgemm batch dimension");
    }
    output_shape[output_shape.size() - 2] = arg0_shape_tmp[arg0_shape_tmp.size() - 2];  // M
    output_shape[output_shape.size() - 1] = arg1_shape_tmp[arg1_shape_tmp.size() - 1];  // N

    // removing the temporary axes from originally 1D tensors.
    if (arg0_shape.rank().get_length() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 2);
    }
    if (arg1_shape.rank().get_length() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 1);
    }
    return output_shape;
}

} // namespace op
} // namespace snippets
} // namespace ngraph
