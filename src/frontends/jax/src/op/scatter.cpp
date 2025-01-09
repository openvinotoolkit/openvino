// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include <cstdint>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "utils.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"


namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

std::vector<int64_t> find_missing_dimensions(const int64_t total_dims, const std::vector<int64_t>& present_dims) {
    std::vector<bool> is_present(total_dims, false);
    
    for (int64_t dim : present_dims) {
        if (dim >= 0 && dim < total_dims) {
            is_present[dim] = true;
        }
    }
    
    std::vector<int64_t> missing_dims;
    for (int64_t i = 0; i < total_dims; ++i) {
        if (!is_present[i]) {
            missing_dims.push_back(i);
        }
    }

    return missing_dims;
}


// Computes the flat indices for scatter dimensions using stride-based indexing
// - total_prod_dims: Total product of dimensions in the tensor
// - scatter_dims_list: List of scatter dimensions
// - cumulative_prod_dims: Cumulative product of dimensions used to calculate strides
std::vector<int64_t> compute_scatter_flat_indices(size_t total_prod_dims, const std::vector<int64_t>& scatter_dims_list, 
                                                  const std::vector<size_t>& cumulative_prod_dims) {
    std::vector<int64_t> flat_indices;
    // Iteratively compute flat indices based on scatter dimensions
    for (int64_t scatter_dim : scatter_dims_list) {
        int64_t end = scatter_dim == 0 ? total_prod_dims : cumulative_prod_dims[scatter_dim - 1];
        int64_t step = cumulative_prod_dims[scatter_dim];

        std::vector<int64_t> temp_indices = std::move(flat_indices);
        flat_indices.clear();

        if (!temp_indices.empty()) {
            // Expand indices for each scatter dimension
            for (int64_t base_index : temp_indices) {
                for (int64_t i = 0; i < end; i += step) {
                    flat_indices.push_back(base_index + i);
                }
            }
        } else {
            // Initialize flat indices if this is the first dimension
            for (int64_t i = 0; i < end; i += step) {
                flat_indices.push_back(i);
            }
        }
    }
    return flat_indices;
}

OutputVector translate_scatter(const NodeContext& context){
    num_inputs_check(context, 3, 3);
    Output<Node> input = context.get_input(0);
    Output<Node> scatter_indices = context.get_input(1);
    Output<Node> updates = context.get_input(2);
    
    auto dimension_numbers = context.const_named_param<std::vector<std::vector<int64_t>>>("dimension_numbers");
    
    JAX_OP_CONVERSION_CHECK(dimension_numbers.size() == 3,
                            "Internal error: dimension_numbers must have 3 vectors but actually got ",
                            dimension_numbers.size());

    auto update_window_dimensions = dimension_numbers[0];
    auto inserted_window_dimensions = dimension_numbers[1];
    auto scatter_dims_to_operand_dims = dimension_numbers[2];

    int64_t num_update_window_dims = update_window_dimensions.size();
    int64_t num_inserted_window_dims = inserted_window_dimensions.size();
    int64_t num_scatter_operand_dims = scatter_dims_to_operand_dims.size();

    Shape input_shape = input.get_shape();
    std::vector<int64_t> input_dims(input_shape.begin(), input_shape.end());
    int64_t num_input_dims = input_dims.size();
    
    JAX_OP_CONVERSION_CHECK(num_update_window_dims + num_inserted_window_dims == num_input_dims, 
                            "Dimension Error : operand.rank: ", num_input_dims, 
                            " must equal the sum of update_window_dims.size: ", num_update_window_dims, 
                            " and inserted_window_dims.size: ", num_inserted_window_dims);

    Shape scatter_indices_shape = scatter_indices.get_shape();
    std::vector<int64_t> scatter_indices_dims(scatter_indices_shape.begin(), scatter_indices_shape.end());
    int64_t num_scatter_indices_dims = scatter_indices_dims.size();

    Shape updates_shape = updates.get_shape();
    std::vector<int64_t> updates_dims(updates_shape.begin(), updates_shape.end());
    int64_t num_updates_dims = updates_dims.size();

    JAX_OP_CONVERSION_CHECK(num_update_window_dims + num_scatter_indices_dims - 1 == num_updates_dims, "Dimension Error : updates array must be of rank update_window_dims.size + scatter_indices.rank - 1 but got update rank ",
                            num_updates_dims, " instead of ", num_update_window_dims + num_scatter_indices_dims - 1);
    JAX_OP_CONVERSION_CHECK(std::is_sorted(update_window_dimensions.begin(), update_window_dimensions.end()), "Internal Error : update_window_dims must be sorted");
    JAX_OP_CONVERSION_CHECK(std::is_sorted(inserted_window_dimensions.begin(), inserted_window_dimensions.end()), "Internal Error : inserted_window_dims must be sorted");
    JAX_OP_CONVERSION_CHECK(num_updates_dims > *std::max_element(update_window_dimensions.begin(), update_window_dimensions.end()), "Internal Error : update_window_dims values must be in the range [0, updates.rank), but got ",
                            *std::max_element(update_window_dimensions.begin(), update_window_dimensions.end()));
    JAX_OP_CONVERSION_CHECK(num_input_dims > *std::max_element(inserted_window_dimensions.begin(), inserted_window_dimensions.end()), "Internal Error : inserted_window_dims values must be in the range [0, operand.rank), but got ",
                            *std::max_element(inserted_window_dimensions.begin(), inserted_window_dimensions.end()));
    JAX_OP_CONVERSION_CHECK(num_input_dims > *std::max_element(scatter_dims_to_operand_dims.begin(), scatter_dims_to_operand_dims.end()), "Internal Error : scatter_dims_to_operand_dims values must be in the range [0, operand.rank), but got ",
                             *std::max_element(scatter_dims_to_operand_dims.begin(), scatter_dims_to_operand_dims.end()));
    
    // Precompute cumulative products for input dimensions for stride calculation
    std::vector<size_t> cumulative_prod_dims;
    size_t total_prod = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<size_t>());
    size_t cumulative_prod = total_prod;

    for(int64_t i=0; i<num_input_dims; i++){
        cumulative_prod /= input_dims[i];
        cumulative_prod_dims.push_back(cumulative_prod);
    }

    std::vector<size_t> scatter_indices_stride(num_scatter_operand_dims);
    for(int64_t i=0; i<num_scatter_operand_dims; i++)
        scatter_indices_stride[i] = cumulative_prod_dims[scatter_dims_to_operand_dims[i]];

    auto strides_tensor = std::make_shared<v0::Constant>(ov::element::i64, Shape{scatter_indices_stride.size(), 1}, scatter_indices_stride);
    
    if (scatter_indices.get_element_type() != ov::element::i64) 
        scatter_indices = std::make_shared<ov::op::v0::Convert>(scatter_indices, ov::element::i64);
    
    // Flatten scatter indices
    auto flattened_scatter_indices = std::make_shared<v0::MatMul>(scatter_indices, strides_tensor);

    // Handles the case where scatter dimensions match input dimensions
    if(num_scatter_operand_dims == num_input_dims){
        auto flatten_shape = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, -1);
        auto flattened_updates = std::make_shared<v1::Reshape>(updates, flatten_shape, false);
        auto flattened_input = std::make_shared<v1::Reshape>(input, flatten_shape, false);

        auto scatter_elements_update = std::make_shared<v3::ScatterElementsUpdate>(flattened_input, flattened_scatter_indices, flattened_updates, flatten_shape);
        auto res_shape = std::make_shared<v0::Constant>(ov::element::i32, Shape{input_dims.size()}, input_dims);
        return {std::make_shared<v1::Reshape>(scatter_elements_update, res_shape, false)};
    }

    // Handles cases with extra missing scatter dimensions
    std::vector<int64_t> scatter_missing_dims = find_missing_dimensions(num_input_dims, scatter_dims_to_operand_dims);
    std::vector<int64_t> scatter_flat_indices = compute_scatter_flat_indices(total_prod, scatter_missing_dims, cumulative_prod_dims);

    // Extend scatter indices shape to include missing dimensions
    scatter_indices_dims.back() = scatter_flat_indices.size();
    auto broadcast_shape_tensor = std::make_shared<v0::Constant>(ov::element::i64, Shape{scatter_indices_dims.size()}, scatter_indices_dims);
    
    // Broadcast indices for both scatter and missing scatter dimensions
    auto broadcasted_flattened_indices = std::make_shared<v3::Broadcast>(flattened_scatter_indices, broadcast_shape_tensor, BroadcastType::NUMPY);  
    auto scatter_flat_indices_tensor = std::make_shared<v0::Constant>(ov::element::i64, Shape{scatter_flat_indices.size()}, scatter_flat_indices); 
    auto broadcasted_flat_scatter_indices = std::make_shared<v3::Broadcast>(scatter_flat_indices_tensor, broadcast_shape_tensor, BroadcastType::NUMPY);
    
    // Combine broadcasted indices
    auto combined_scatter_indices = std::make_shared<v1::Add>(broadcasted_flattened_indices, broadcasted_flat_scatter_indices);
    
    // Flatten input, updates and scatter indices tensors for scatter operation
    auto flatten_shape = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, -1);
    auto flattened_updates = std::make_shared<v1::Reshape>(updates, flatten_shape, false);
    auto flattened_input = std::make_shared<v1::Reshape>(input, flatten_shape, false);
    auto flattened_combined_indices = std::make_shared<v1::Reshape>(combined_scatter_indices, flatten_shape, false);

    auto scatter_elements_update = std::make_shared<v3::ScatterElementsUpdate>(flattened_input, flattened_combined_indices, flattened_updates, flatten_shape);
    
    // Reshape the result back to original input dimensions
    auto res_shape = std::make_shared<v0::Constant>(ov::element::i32, Shape{input_dims.size()}, input_dims);
    Output<Node> res = std::make_shared<v1::Reshape>(scatter_elements_update, res_shape, false);
    return {res};

};

}
}
}
}



