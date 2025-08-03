// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_decomposition.hpp"

#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {

GridSampleDecomposition::GridSampleDecomposition() {
    auto grid_sample_pattern = ov::pass::pattern::wrap_type<ov::op::v9::GridSample>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto grid_sample = ov::as_type_ptr<ov::op::v9::GridSample>(m.get_match_root());
        if (!grid_sample) {
            return false;
        }

        // Currently only support bilinear interpolation with border padding
        const auto& attrs = grid_sample->get_attributes();
        if (attrs.mode != ov::op::v9::GridSample::InterpolationMode::BILINEAR ||
            attrs.padding_mode != ov::op::v9::GridSample::PaddingMode::BORDER) {
            return false;
        }

        auto data = grid_sample->input_value(0);  // [N, C, H_in, W_in]
        auto grid = grid_sample->input_value(1);  // [N, H_out, W_out, 2]
        
        const auto data_shape = data.get_partial_shape();
        const auto grid_shape = grid.get_partial_shape();
        
        // Only support 4D data and grid tensors
        if (!data_shape.is_static() || !grid_shape.is_static() ||
            data_shape.rank().get_length() != 4 || grid_shape.rank().get_length() != 4) {
            return false;
        }

        const auto element_type = data.get_element_type();
        
        // Get shape components
        auto data_shape_node = std::make_shared<ov::op::v3::ShapeOf>(data);
        auto h_in = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                                                          ov::op::v0::Constant::create(ov::element::i32, {1}, {2}),
                                                          ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        auto w_in = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                                                          ov::op::v0::Constant::create(ov::element::i32, {1}, {3}),
                                                          ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        
        // Convert to float for calculations
        h_in = std::make_shared<ov::op::v0::Convert>(h_in, element_type);
        w_in = std::make_shared<ov::op::v0::Convert>(w_in, element_type);
        
        // Constants
        auto const_0 = ov::op::v0::Constant::create(element_type, {}, {0.0f});
        auto const_1 = ov::op::v0::Constant::create(element_type, {}, {1.0f});
        auto const_2 = ov::op::v0::Constant::create(element_type, {}, {2.0f});
        
        // Split grid into x and y coordinates
        auto split_axis = ov::op::v0::Constant::create(ov::element::i32, {}, {3});
        auto grid_split = std::make_shared<ov::op::v1::Split>(grid, split_axis, 2);
        auto grid_x = grid_split->output(0);  // [N, H_out, W_out, 1]
        auto grid_y = grid_split->output(1);  // [N, H_out, W_out, 1]
        
        // Squeeze last dimension
        auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {3});
        grid_x = std::make_shared<ov::op::v0::Squeeze>(grid_x, squeeze_axis);  // [N, H_out, W_out]
        grid_y = std::make_shared<ov::op::v0::Squeeze>(grid_y, squeeze_axis);  // [N, H_out, W_out]
        
        // Denormalize coordinates from [-1, 1] to pixel coordinates
        // x = ((grid_x + 1) * (W_in - 1)) / 2
        // y = ((grid_y + 1) * (H_in - 1)) / 2
        auto w_in_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in, const_1);
        auto h_in_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in, const_1);
        
        auto x_plus_1 = std::make_shared<ov::op::v1::Add>(grid_x, const_1);
        auto y_plus_1 = std::make_shared<ov::op::v1::Add>(grid_y, const_1);
        
        auto x_denorm = std::make_shared<ov::op::v1::Multiply>(x_plus_1, w_in_minus_1);
        auto y_denorm = std::make_shared<ov::op::v1::Multiply>(y_plus_1, h_in_minus_1);
        
        auto x = std::make_shared<ov::op::v1::Divide>(x_denorm, const_2);
        auto y = std::make_shared<ov::op::v1::Divide>(y_denorm, const_2);
        
        // Find neighboring coordinates
        auto x0 = std::make_shared<ov::op::v0::Floor>(x);
        auto y0 = std::make_shared<ov::op::v0::Floor>(y);
        auto x1 = std::make_shared<ov::op::v1::Add>(x0, const_1);
        auto y1 = std::make_shared<ov::op::v1::Add>(y0, const_1);
        
        // Clamp coordinates to image boundaries
        x0 = std::make_shared<ov::op::v0::Clamp>(x0, 0.0f, std::numeric_limits<float>::max());
        x1 = std::make_shared<ov::op::v0::Clamp>(x1, 0.0f, std::numeric_limits<float>::max());
        y0 = std::make_shared<ov::op::v0::Clamp>(y0, 0.0f, std::numeric_limits<float>::max());
        y1 = std::make_shared<ov::op::v0::Clamp>(y1, 0.0f, std::numeric_limits<float>::max());
        
        x0 = std::make_shared<ov::op::v1::Minimum>(x0, w_in_minus_1);
        x1 = std::make_shared<ov::op::v1::Minimum>(x1, w_in_minus_1);
        y0 = std::make_shared<ov::op::v1::Minimum>(y0, h_in_minus_1);
        y1 = std::make_shared<ov::op::v1::Minimum>(y1, h_in_minus_1);
        
        // Convert to indices
        auto x0_idx = std::make_shared<ov::op::v0::Convert>(x0, ov::element::i32);
        auto x1_idx = std::make_shared<ov::op::v0::Convert>(x1, ov::element::i32);
        auto y0_idx = std::make_shared<ov::op::v0::Convert>(y0, ov::element::i32);
        auto y1_idx = std::make_shared<ov::op::v0::Convert>(y1, ov::element::i32);
        
        // Create batch and channel indices
        auto grid_shape_node = std::make_shared<ov::op::v3::ShapeOf>(grid);
        auto batch_size = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
                                                               ov::op::v0::Constant::create(ov::element::i32, {1}, {0}),
                                                               ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        auto h_out = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
                                                          ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                                                          ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        auto w_out = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
                                                          ov::op::v0::Constant::create(ov::element::i32, {1}, {2}),
                                                          ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        
        auto batch_range = std::make_shared<ov::op::v4::Range>(ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                                                               batch_size,
                                                               ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                                                               ov::element::i32);
        
        // Reshape batch indices to [N, 1, 1]
        auto batch_shape = std::make_shared<ov::op::v0::Concat>(std::vector<Output<Node>>{
            batch_size,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, {1}, {1})}, 0);
        auto batch_indices = std::make_shared<ov::op::v1::Reshape>(batch_range, batch_shape, false);
        
        // Broadcast batch indices to [N, H_out, W_out]
        auto broadcast_shape = std::make_shared<ov::op::v0::Concat>(std::vector<Output<Node>>{
            batch_size, h_out, w_out}, 0);
        auto batch_broadcast = std::make_shared<ov::op::v3::Broadcast>(batch_indices, broadcast_shape);
        
        // Get number of channels
        auto channels = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                                                             ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                                                             ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        
        // Initialize result as zeros
        auto output_shape = std::make_shared<ov::op::v0::Concat>(std::vector<Output<Node>>{
            batch_size, channels, h_out, w_out}, 0);
        auto result = ov::op::v0::Constant::create(element_type, {}, {0.0f});
        result = std::make_shared<ov::op::v3::Broadcast>(result, output_shape);
        
        // For each channel, perform bilinear interpolation
        // This is simplified - in real implementation we'd need to iterate over channels
        // For now, we'll create indices for gathering
        
        // Stack indices for GatherND: [batch_idx, channel_idx, y_idx, x_idx]
        // We'll need to expand dimensions and create proper indices
        auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});
        
        // Helper function to create indices for GatherND
        auto create_indices = [&](const Output<Node>& y_idx, const Output<Node>& x_idx) -> Output<Node> {
            // Stack [batch, channel=0, y, x] indices
            auto y_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(y_idx, unsqueeze_axis);
            auto x_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(x_idx, unsqueeze_axis);
            auto batch_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(batch_broadcast, unsqueeze_axis);
            
            // For simplicity, we'll process all channels at once by reshaping
            return std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{batch_unsqueeze, y_unsqueeze, x_unsqueeze}, -1);
        };
        
        // Create gather indices for the four corner points
        // We'll use GatherND with indices of shape [N, H_out, W_out, 4]
        // where the last dimension contains [batch_idx, channel_idx, y_idx, x_idx]
        
        // Helper function to create full indices for GatherND
        auto create_gather_indices = [&](const Output<Node>& y_idx, const Output<Node>& x_idx) -> Output<Node> {
            // Expand dimensions to [N, H_out, W_out, 1]
            auto batch_exp = std::make_shared<ov::op::v0::Unsqueeze>(batch_broadcast, unsqueeze_axis);
            auto y_exp = std::make_shared<ov::op::v0::Unsqueeze>(y_idx, unsqueeze_axis);
            auto x_exp = std::make_shared<ov::op::v0::Unsqueeze>(x_idx, unsqueeze_axis);
            
            // We'll gather all channels at once by using a more efficient approach
            // For each spatial location, we gather a slice along the channel dimension
            // Create indices [batch, :, y, x] which will gather all channels
            return std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{batch_exp, y_exp, x_exp}, -1);
        };
        
        auto indices_00 = create_gather_indices(y0_idx, x0_idx);
        auto indices_01 = create_gather_indices(y0_idx, x1_idx);
        auto indices_10 = create_gather_indices(y1_idx, x0_idx);
        auto indices_11 = create_gather_indices(y1_idx, x1_idx);
        
        // Use a simplified approach: for each corner, extract the values across all channels
        // We'll process each channel separately using a loop-like structure
        
        // First, let's get the values for a single channel and then extend to all channels
        auto gather_values = [&](const Output<Node>& y_idx, const Output<Node>& x_idx) -> Output<Node> {
            // Create a channels range [0, 1, ..., C-1]
            auto channels_range = std::make_shared<ov::op::v4::Range>(
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                channels,
                ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                ov::element::i32);
            
            // Reshape to [1, C, 1, 1] for broadcasting
            auto channels_shape = std::make_shared<ov::op::v0::Concat>(std::vector<Output<Node>>{
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                channels,
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1})}, 0);
            auto channels_reshaped = std::make_shared<ov::op::v1::Reshape>(channels_range, channels_shape, false);
            
            // Broadcast to [N, C, H_out, W_out]
            auto target_shape = output_shape;
            auto channels_broadcast = std::make_shared<ov::op::v3::Broadcast>(channels_reshaped, target_shape);
            
            // Prepare batch indices broadcast to [N, C, H_out, W_out]
            auto batch_4d = std::make_shared<ov::op::v0::Unsqueeze>(batch_broadcast,
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
            batch_4d = std::make_shared<ov::op::v3::Broadcast>(batch_4d, target_shape);
            
            // Prepare y and x indices to [N, C, H_out, W_out]
            auto y_idx_4d = std::make_shared<ov::op::v0::Unsqueeze>(y_idx,
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
            y_idx_4d = std::make_shared<ov::op::v3::Broadcast>(y_idx_4d, target_shape);
            
            auto x_idx_4d = std::make_shared<ov::op::v0::Unsqueeze>(x_idx,
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
            x_idx_4d = std::make_shared<ov::op::v3::Broadcast>(x_idx_4d, target_shape);
            
            // Stack indices for GatherND
            auto batch_unsq = std::make_shared<ov::op::v0::Unsqueeze>(batch_4d, unsqueeze_axis);
            auto channels_unsq = std::make_shared<ov::op::v0::Unsqueeze>(channels_broadcast, unsqueeze_axis);
            auto y_unsq = std::make_shared<ov::op::v0::Unsqueeze>(y_idx_4d, unsqueeze_axis);
            auto x_unsq = std::make_shared<ov::op::v0::Unsqueeze>(x_idx_4d, unsqueeze_axis);
            
            auto full_indices = std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{batch_unsq, channels_unsq, y_unsq, x_unsq}, -1);
            
            return std::make_shared<ov::op::v8::GatherND>(data, full_indices);
        };
        
        // Extract values at four corners
        auto v_00 = gather_values(y0_idx, x0_idx);  // [N, C, H_out, W_out]
        auto v_01 = gather_values(y0_idx, x1_idx);
        auto v_10 = gather_values(y1_idx, x0_idx);
        auto v_11 = gather_values(y1_idx, x1_idx);
        
        // Calculate interpolation weights
        auto x1_minus_x = std::make_shared<ov::op::v1::Subtract>(x1, x);
        auto x_minus_x0 = std::make_shared<ov::op::v1::Subtract>(x, x0);
        auto y1_minus_y = std::make_shared<ov::op::v1::Subtract>(y1, y);
        auto y_minus_y0 = std::make_shared<ov::op::v1::Subtract>(y, y0);
        
        // Expand weights to match data dimensions [N, 1, H_out, W_out]
        auto expand_weight = [&](const Output<Node>& weight) -> Output<Node> {
            return std::make_shared<ov::op::v0::Unsqueeze>(weight, 
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
        };
        
        x1_minus_x = expand_weight(x1_minus_x);
        x_minus_x0 = expand_weight(x_minus_x0);
        y1_minus_y = expand_weight(y1_minus_y);
        y_minus_y0 = expand_weight(y_minus_y0);
        
        // Compute bilinear interpolation
        // result = w_00 * v_00 + w_01 * v_01 + w_10 * v_10 + w_11 * v_11
        auto w_00 = std::make_shared<ov::op::v1::Multiply>(x1_minus_x, y1_minus_y);
        auto w_01 = std::make_shared<ov::op::v1::Multiply>(x_minus_x0, y1_minus_y);
        auto w_10 = std::make_shared<ov::op::v1::Multiply>(x1_minus_x, y_minus_y0);
        auto w_11 = std::make_shared<ov::op::v1::Multiply>(x_minus_x0, y_minus_y0);
        
        auto term_00 = std::make_shared<ov::op::v1::Multiply>(w_00, v_00);
        auto term_01 = std::make_shared<ov::op::v1::Multiply>(w_01, v_01);
        auto term_10 = std::make_shared<ov::op::v1::Multiply>(w_10, v_10);
        auto term_11 = std::make_shared<ov::op::v1::Multiply>(w_11, v_11);
        
        auto sum_0 = std::make_shared<ov::op::v1::Add>(term_00, term_01);
        auto sum_1 = std::make_shared<ov::op::v1::Add>(term_10, term_11);
        result = std::make_shared<ov::op::v1::Add>(sum_0, sum_1);
        
        // Copy runtime info and replace node
        result->set_friendly_name(grid_sample->get_friendly_name());
        ov::copy_runtime_info(grid_sample, 
            {x, y, x0, x1, y0, y1, v_00, v_01, v_10, v_11, 
             w_00, w_01, w_10, w_11, term_00, term_01, term_10, term_11, 
             sum_0, sum_1, result});
        ov::replace_node(grid_sample, result);
        
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(grid_sample_pattern, "GridSampleDecomposition");
    register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov