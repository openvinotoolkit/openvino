// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_decomposition.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov {
namespace intel_cpu {

GridSampleDecomposition::GridSampleDecomposition() {
    auto grid_sample_pattern = ov::pass::pattern::wrap_type<ov::op::v9::GridSample>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto grid_sample = ov::as_type_ptr<ov::op::v9::GridSample>(m.get_match_root());
        if (!grid_sample) {
            return false;
        }

        const auto& attrs = grid_sample->get_attributes();
        
        // Support all interpolation modes and padding modes
        const bool is_bilinear = attrs.mode == ov::op::v9::GridSample::InterpolationMode::BILINEAR;
        const bool is_nearest = attrs.mode == ov::op::v9::GridSample::InterpolationMode::NEAREST;
        const bool is_bicubic = attrs.mode == ov::op::v9::GridSample::InterpolationMode::BICUBIC;
        
        const bool is_border = attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::BORDER;
        const bool is_zeros = attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::ZEROS;
        const bool is_reflection = attrs.padding_mode == ov::op::v9::GridSample::PaddingMode::REFLECTION;

        auto data = grid_sample->input_value(0);  // [N, C, H_in, W_in]
        auto grid = grid_sample->input_value(1);  // [N, H_out, W_out, 2]
        
        const auto data_shape = data.get_partial_shape();
        const auto grid_shape = grid.get_partial_shape();
        
        // Support both static and dynamic shapes, but must be 4D tensors
        if (data_shape.rank().get_length() != 4 || grid_shape.rank().get_length() != 4) {
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
        auto h_in_f = std::make_shared<ov::op::v0::Convert>(h_in, element_type);
        auto w_in_f = std::make_shared<ov::op::v0::Convert>(w_in, element_type);
        
        // Constants
        auto const_1 = ov::op::v0::Constant::create(element_type, {}, {1.0f});
        auto const_2 = ov::op::v0::Constant::create(element_type, {}, {2.0f});
        
        // Convert grid to data element type if needed
        auto grid_converted = grid;
        if (grid.get_element_type() != element_type) {
            grid_converted = std::make_shared<ov::op::v0::Convert>(grid, element_type);
        }
        
        // Split grid into x and y coordinates [N, H_out, W_out, 1] each
        auto split_axis = ov::op::v0::Constant::create(ov::element::i32, {}, {3});
        auto grid_split = std::make_shared<ov::op::v1::Split>(grid_converted, split_axis, 2);
        auto grid_x = grid_split->output(0);
        auto grid_y = grid_split->output(1);
        
        // Squeeze last dimension to get [N, H_out, W_out]
        auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {3});
        auto grid_x_sq = std::make_shared<ov::op::v0::Squeeze>(grid_x, squeeze_axis);
        auto grid_y_sq = std::make_shared<ov::op::v0::Squeeze>(grid_y, squeeze_axis);
        
        // Denormalize coordinates from [-1, 1] to pixel coordinates
        // If align_corners=True:  x = (grid_x + 1) * (W_in - 1) / 2
        // If align_corners=False: x = ((grid_x + 1) * W_in - 1) / 2
        auto w_in_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
        auto h_in_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
        
        auto x_plus_1 = std::make_shared<ov::op::v1::Add>(grid_x_sq, const_1);
        auto y_plus_1 = std::make_shared<ov::op::v1::Add>(grid_y_sq, const_1);
        
        std::shared_ptr<ov::Node> x, y;
        if (attrs.align_corners) {
            // align_corners=True: x = (grid_x + 1) * (W_in - 1) / 2
            auto x_denorm = std::make_shared<ov::op::v1::Multiply>(x_plus_1, w_in_minus_1);
            auto y_denorm = std::make_shared<ov::op::v1::Multiply>(y_plus_1, h_in_minus_1);
            x = std::make_shared<ov::op::v1::Divide>(x_denorm, const_2);
            y = std::make_shared<ov::op::v1::Divide>(y_denorm, const_2);
        } else {
            // align_corners=False: x = ((grid_x + 1) * W_in - 1) / 2
            auto x_denorm = std::make_shared<ov::op::v1::Multiply>(x_plus_1, w_in_f);
            auto y_denorm = std::make_shared<ov::op::v1::Multiply>(y_plus_1, h_in_f);
            auto x_denorm_minus_1 = std::make_shared<ov::op::v1::Subtract>(x_denorm, const_1);
            auto y_denorm_minus_1 = std::make_shared<ov::op::v1::Subtract>(y_denorm, const_1);
            x = std::make_shared<ov::op::v1::Divide>(x_denorm_minus_1, const_2);
            y = std::make_shared<ov::op::v1::Divide>(y_denorm_minus_1, const_2);
        }
        
        // For nearest interpolation, round to nearest coordinate
        std::shared_ptr<ov::Node> x_idx, y_idx;
        if (is_nearest) {
            // Use Round operation to match std::lrint behavior exactly
            // The align_corners difference is already handled in the denormalization
            x_idx = std::make_shared<ov::op::v5::Round>(x, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
            y_idx = std::make_shared<ov::op::v5::Round>(y, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        }
        
        // Find neighboring coordinates for bilinear/bicubic
        auto x0 = std::make_shared<ov::op::v0::Floor>(x);
        auto y0 = std::make_shared<ov::op::v0::Floor>(y);
        auto x1 = std::make_shared<ov::op::v1::Add>(x0, const_1);
        auto y1 = std::make_shared<ov::op::v1::Add>(y0, const_1);
        
        // For bicubic, we need more neighbors
        std::shared_ptr<ov::Node> x_1, y_1, x2, y2;
        if (is_bicubic) {
            x_1 = std::make_shared<ov::op::v1::Subtract>(x0, const_1);
            y_1 = std::make_shared<ov::op::v1::Subtract>(y0, const_1);
            x2 = std::make_shared<ov::op::v1::Add>(x1, const_1);
            y2 = std::make_shared<ov::op::v1::Add>(y1, const_1);
        }
        
        // Apply padding mode to coordinates
        auto zero = ov::op::v0::Constant::create(element_type, {}, {0.0f});
        std::shared_ptr<ov::Node> x0_bounded, x1_bounded, y0_bounded, y1_bounded;
        std::shared_ptr<ov::Node> x_idx_bounded, y_idx_bounded;
        
        auto apply_padding = [&](const std::shared_ptr<ov::Node>& coord, 
                                const std::shared_ptr<ov::Node>& size_minus_1,
                                const std::string& name) -> std::shared_ptr<ov::Node> {
            if (is_border || is_zeros) {
                // Clamp to [0, size-1]
                auto clamped = std::make_shared<ov::op::v0::Clamp>(coord, 0.0f, std::numeric_limits<float>::max());
                return std::make_shared<ov::op::v1::Minimum>(clamped, size_minus_1);
            } else if (is_reflection) {
                auto size = std::make_shared<ov::op::v1::Add>(size_minus_1, const_1);
                
                if (attrs.align_corners) {
                    // align_corners=True: period = 2 * (size - 1), matching reference exactly
                    auto size_eq_1 = std::make_shared<ov::op::v1::Equal>(size, const_1);
                    auto one_val = ov::op::v0::Constant::create(element_type, {}, {1.0f});
                    auto period = std::make_shared<ov::op::v1::Select>(size_eq_1, one_val, 
                                    std::make_shared<ov::op::v1::Multiply>(size_minus_1, const_2));
                    
                    // Use abs(coord) % period just like reference
                    auto abs_coord = std::make_shared<ov::op::v0::Abs>(coord);
                    auto remainder = std::make_shared<ov::op::v1::FloorMod>(abs_coord, period);
                    
                    // If remainder >= size, reflect: period - remainder (NOT period - 1 - remainder)
                    auto need_reflect = std::make_shared<ov::op::v1::GreaterEqual>(remainder, size);
                    auto reflected = std::make_shared<ov::op::v1::Subtract>(period, remainder);
                    auto result = std::make_shared<ov::op::v1::Select>(need_reflect, reflected, remainder);
                    
                    return std::make_shared<ov::op::v1::Select>(size_eq_1, zero, result);
                } else {
                    // align_corners=False: period = 2 * size, matching reference exactly
                    auto period = std::make_shared<ov::op::v1::Multiply>(size, const_2);
                    
                    // Convert to int32 for proper modulo with negative numbers
                    auto coord_int = std::make_shared<ov::op::v0::Convert>(coord, ov::element::i32);
                    auto period_int = std::make_shared<ov::op::v0::Convert>(period, ov::element::i32);
                    auto size_int = std::make_shared<ov::op::v0::Convert>(size, ov::element::i32);
                    
                    // Exactly like reference: (coord % period + period) % period
                    auto mod1 = std::make_shared<ov::op::v1::FloorMod>(coord_int, period_int);
                    auto plus_period = std::make_shared<ov::op::v1::Add>(mod1, period_int);
                    auto mod2 = std::make_shared<ov::op::v1::FloorMod>(plus_period, period_int);
                    
                    // Exactly like reference: period - 1 - result (not period - result)
                    auto need_reflect = std::make_shared<ov::op::v1::GreaterEqual>(mod2, size_int);
                    auto period_minus_1 = std::make_shared<ov::op::v1::Subtract>(period_int, 
                                            ov::op::v0::Constant::create(ov::element::i32, {}, {1}));
                    auto reflected = std::make_shared<ov::op::v1::Subtract>(period_minus_1, mod2);
                    auto result_int = std::make_shared<ov::op::v1::Select>(need_reflect, reflected, mod2);
                    
                    return std::make_shared<ov::op::v0::Convert>(result_int, coord->get_element_type());
                }
            }
            return coord;
        };
        
        if (is_nearest) {
            // For reflection padding with nearest neighbor, we need integer coordinates
            // The Round operation gives us float, but reflection padding needs proper integer modulo
            if (is_reflection) {
                // Convert rounded float to int32 first to match std::lrint behavior exactly
                auto x_idx_int = std::make_shared<ov::op::v0::Convert>(x_idx, ov::element::i32);
                auto y_idx_int = std::make_shared<ov::op::v0::Convert>(y_idx, ov::element::i32);
                auto x_idx_float = std::make_shared<ov::op::v0::Convert>(x_idx_int, element_type);
                auto y_idx_float = std::make_shared<ov::op::v0::Convert>(y_idx_int, element_type);
                x_idx_bounded = apply_padding(x_idx_float, w_in_minus_1, "x_idx");
                y_idx_bounded = apply_padding(y_idx_float, h_in_minus_1, "y_idx");
            } else {
                x_idx_bounded = apply_padding(x_idx, w_in_minus_1, "x_idx");
                y_idx_bounded = apply_padding(y_idx, h_in_minus_1, "y_idx");
            }
        }
        
        // For zeros padding, we need to clamp coordinates but not apply border padding
        // The mask will handle out-of-bounds values
        if (is_zeros) {
            // Only clamp to valid range, don't apply padding
            auto clamp_coord = [&](const std::shared_ptr<ov::Node>& coord, 
                                  const std::shared_ptr<ov::Node>& size_minus_1) -> std::shared_ptr<ov::Node> {
                auto clamped = std::make_shared<ov::op::v0::Clamp>(coord, 0.0f, std::numeric_limits<float>::max());
                return std::make_shared<ov::op::v1::Minimum>(clamped, size_minus_1);
            };
            
            x0_bounded = clamp_coord(x0, w_in_minus_1);
            x1_bounded = clamp_coord(x1, w_in_minus_1);
            y0_bounded = clamp_coord(y0, h_in_minus_1);
            y1_bounded = clamp_coord(y1, h_in_minus_1);
        } else {
            x0_bounded = apply_padding(x0, w_in_minus_1, "x0");
            x1_bounded = apply_padding(x1, w_in_minus_1, "x1");
            y0_bounded = apply_padding(y0, h_in_minus_1, "y0");
            y1_bounded = apply_padding(y1, h_in_minus_1, "y1");
        }
        
        std::shared_ptr<ov::Node> x_1_bounded, y_1_bounded, x2_bounded, y2_bounded;
        if (is_bicubic) {
            if (is_zeros) {
                // For zeros padding, just clamp the extended coordinates
                auto clamp_coord = [&](const std::shared_ptr<ov::Node>& coord, 
                                      const std::shared_ptr<ov::Node>& size_minus_1) -> std::shared_ptr<ov::Node> {
                    auto clamped = std::make_shared<ov::op::v0::Clamp>(coord, 0.0f, std::numeric_limits<float>::max());
                    return std::make_shared<ov::op::v1::Minimum>(clamped, size_minus_1);
                };
                x_1_bounded = clamp_coord(x_1, w_in_minus_1);
                y_1_bounded = clamp_coord(y_1, h_in_minus_1);
                x2_bounded = clamp_coord(x2, w_in_minus_1);
                y2_bounded = clamp_coord(y2, h_in_minus_1);
            } else {
                x_1_bounded = apply_padding(x_1, w_in_minus_1, "x_1");
                y_1_bounded = apply_padding(y_1, h_in_minus_1, "y_1");
                x2_bounded = apply_padding(x2, w_in_minus_1, "x2");
                y2_bounded = apply_padding(y2, h_in_minus_1, "y2");
            }
        }
        
        // Convert coordinates to int32 for indexing
        auto x0_int = std::make_shared<ov::op::v0::Convert>(x0_bounded, ov::element::i32);
        auto x1_int = std::make_shared<ov::op::v0::Convert>(x1_bounded, ov::element::i32);
        auto y0_int = std::make_shared<ov::op::v0::Convert>(y0_bounded, ov::element::i32);
        auto y1_int = std::make_shared<ov::op::v0::Convert>(y1_bounded, ov::element::i32);
        
        // Calculate interpolation weights
        auto x1_minus_x = std::make_shared<ov::op::v1::Subtract>(x1, x);
        auto x_minus_x0 = std::make_shared<ov::op::v1::Subtract>(x, x0);
        auto y1_minus_y = std::make_shared<ov::op::v1::Subtract>(y1, y);
        auto y_minus_y0 = std::make_shared<ov::op::v1::Subtract>(y, y0);
        
        // For zeros padding, we create individual masks for each corner point
        // rather than a global mask for the entire interpolation
        std::shared_ptr<ov::Node> mask_00, mask_01, mask_10, mask_11;
        if (is_zeros && (is_bilinear || is_bicubic)) {
            // Create individual masks for each corner point
            auto x0_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                std::make_shared<ov::op::v1::GreaterEqual>(x0, zero),
                std::make_shared<ov::op::v1::Less>(x0, w_in_f));
            auto x1_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                std::make_shared<ov::op::v1::GreaterEqual>(x1, zero),
                std::make_shared<ov::op::v1::Less>(x1, w_in_f));
            auto y0_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                std::make_shared<ov::op::v1::GreaterEqual>(y0, zero),
                std::make_shared<ov::op::v1::Less>(y0, h_in_f));
            auto y1_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                std::make_shared<ov::op::v1::GreaterEqual>(y1, zero),
                std::make_shared<ov::op::v1::Less>(y1, h_in_f));
            
            // Each corner mask is the combination of its x and y validity
            mask_00 = std::make_shared<ov::op::v0::Convert>(
                std::make_shared<ov::op::v1::LogicalAnd>(x0_valid, y0_valid), element_type);
            mask_01 = std::make_shared<ov::op::v0::Convert>(
                std::make_shared<ov::op::v1::LogicalAnd>(x1_valid, y0_valid), element_type);
            mask_10 = std::make_shared<ov::op::v0::Convert>(
                std::make_shared<ov::op::v1::LogicalAnd>(x0_valid, y1_valid), element_type);
            mask_11 = std::make_shared<ov::op::v0::Convert>(
                std::make_shared<ov::op::v1::LogicalAnd>(x1_valid, y1_valid), element_type);
            
            // Add channel dimension to each mask for [N, 1, H, W] format
            auto unsqueeze_ax = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
            mask_00 = std::make_shared<ov::op::v0::Unsqueeze>(mask_00, unsqueeze_ax);
            mask_01 = std::make_shared<ov::op::v0::Unsqueeze>(mask_01, unsqueeze_ax);
            mask_10 = std::make_shared<ov::op::v0::Unsqueeze>(mask_10, unsqueeze_ax);
            mask_11 = std::make_shared<ov::op::v0::Unsqueeze>(mask_11, unsqueeze_ax);
        }
        
        // For nearest neighbor with zeros padding
        std::shared_ptr<ov::Node> mask_nearest;
        if (is_zeros && is_nearest) {
            auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                std::make_shared<ov::op::v1::GreaterEqual>(x_idx, zero),
                std::make_shared<ov::op::v1::Less>(x_idx, w_in_f));
            auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                std::make_shared<ov::op::v1::GreaterEqual>(y_idx, zero),
                std::make_shared<ov::op::v1::Less>(y_idx, h_in_f));
            auto valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
            mask_nearest = std::make_shared<ov::op::v0::Convert>(valid, element_type);
            // Add channel dimension for [N, 1, H, W] format
            mask_nearest = std::make_shared<ov::op::v0::Unsqueeze>(mask_nearest, ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
        }
        
        // Implement sampling without GatherND using Reshape and element-wise operations
        // Following the user's algorithm: use Add/Subtract/Multiply/Divide/Floor/Clamp/Reshape/Unsqueeze/Scatter
        
        // Get data shape components
        auto n_dim = std::make_shared<ov::op::v8::Gather>(data_shape_node,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {0}),
            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        auto c_dim = std::make_shared<ov::op::v8::Gather>(data_shape_node,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        
        // Get grid output shape
        auto grid_shape_node = std::make_shared<ov::op::v3::ShapeOf>(grid);
        auto h_out = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        auto w_out = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
            ov::op::v0::Constant::create(ov::element::i32, {1}, {2}),
            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
        
        std::shared_ptr<ov::Node> interpolated;
        
        if (is_nearest) {
            // Convert coordinates to int32
            auto x_idx_int = std::make_shared<ov::op::v0::Convert>(x_idx_bounded, ov::element::i32);
            auto y_idx_int = std::make_shared<ov::op::v0::Convert>(y_idx_bounded, ov::element::i32);
            
            // Calculate linear indices: idx = y * W + x
            auto w_in_i32 = std::make_shared<ov::op::v0::Convert>(w_in, ov::element::i32);
            auto linear_idx = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(y_idx_int, w_in_i32), x_idx_int);
            
            // For now, use a simplified approach with Slice operations for each coordinate pair
            // TODO: Full Scatter implementation for complete dynamic support
            
            // Create output tensor with zeros
            auto output_shape = std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{n_dim, c_dim, h_out, w_out}, 0);
            auto zeros_tensor = std::make_shared<ov::op::v0::Constant>(
                element_type, ov::Shape{1}, std::vector<float>{0.0f});
            auto result_tensor = std::make_shared<ov::op::v3::Broadcast>(zeros_tensor, output_shape);
            
            // For static shapes, we can use direct indexing
            // For dynamic shapes, we need the full Scatter approach
            // Currently implementing a basic version that works for both
            
            // Use the coordinates that already have proper padding applied
            auto y_clamped = y_idx_int;
            auto x_clamped = x_idx_int;
            
            // Use element-wise operations to sample from data
            // This is a simplified version - for full dynamic support we need Scatter
            
            // For each output position, we need to find the corresponding input value
            // We'll iterate through spatial positions and use Select operations
            
            // Calculate linear input indices
            auto w_in_i32_near = std::make_shared<ov::op::v0::Convert>(w_in, ov::element::i32);
            auto linear_input_idx = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(y_clamped, w_in_i32_near), x_clamped);
            
            // Реализация без Gather: используем OneHot + Multiply + ReduceSum
            // 1. Привести input к плоскому виду: [N, C, H_in, W_in] -> [N, C, H_in * W_in]
            auto hw_dim = std::make_shared<ov::op::v1::Multiply>(h_in, w_in);
            auto input_flat_shape = std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{n_dim, c_dim, hw_dim}, 0);
            auto input_flat = std::make_shared<ov::op::v1::Reshape>(data, input_flat_shape, false);
            
            // 2. linear_input_idx уже содержит плоские индексы: [N, H_out, W_out]
            // Индексы: y0 * W_in + x0 уже вычислены выше
            
            // 3. Создаем OneHot маску для каждого пикселя
            // OneHot(linear_input_idx, depth=H_in*W_in) -> [N, H_out, W_out, H_in*W_in]
            auto hw_dim_i32 = std::make_shared<ov::op::v0::Convert>(hw_dim, ov::element::i32);
            auto hw_dim_scalar = std::make_shared<ov::op::v0::Squeeze>(hw_dim_i32);
            auto one_hot_mask = std::make_shared<ov::op::v1::OneHot>(
                linear_input_idx,
                hw_dim_scalar,
                ov::op::v0::Constant::create(element_type, {}, {1.0f}),
                ov::op::v0::Constant::create(element_type, {}, {0.0f}),
                -1);
            
            // 4. Расширяем input_flat для broadcasting: [N, C, H_in*W_in] -> [N, C, 1, 1, H_in*W_in]
            auto unsqueeze_ax_1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {2});
            auto unsqueeze_ax_2 = ov::op::v0::Constant::create(ov::element::i32, {1}, {2});
            auto input_expanded = std::make_shared<ov::op::v0::Unsqueeze>(input_flat, unsqueeze_ax_1);
            input_expanded = std::make_shared<ov::op::v0::Unsqueeze>(input_expanded, unsqueeze_ax_2);
            
            // 5. Расширяем one_hot_mask для broadcasting: [N, H_out, W_out, H_in*W_in] -> [N, 1, H_out, W_out, H_in*W_in]
            auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(one_hot_mask, 
                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
            
            // 6. Multiply + ReduceSum для эмуляции gather
            // input_expanded: [N, C, 1, 1, H_in*W_in] * mask_expanded: [N, 1, H_out, W_out, H_in*W_in]
            // -> [N, C, H_out, W_out, H_in*W_in]
            auto multiplied = std::make_shared<ov::op::v1::Multiply>(input_expanded, mask_expanded);
            
            // ReduceSum по последней оси (H_in*W_in)
            auto reduce_axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {4});
            interpolated = std::make_shared<ov::op::v1::ReduceSum>(multiplied, reduce_axis, false);
            
            // For zeros padding apply mask
            if (is_zeros && mask_nearest) {
                interpolated = std::make_shared<ov::op::v1::Multiply>(interpolated, mask_nearest);
            }
            
        } else if (is_bilinear) {
            // Implement bilinear interpolation without GatherND
            // Convert coordinates to int32 for indexing
            auto x0_int = std::make_shared<ov::op::v0::Convert>(x0_bounded, ov::element::i32);
            auto x1_int = std::make_shared<ov::op::v0::Convert>(x1_bounded, ov::element::i32);
            auto y0_int = std::make_shared<ov::op::v0::Convert>(y0_bounded, ov::element::i32);
            auto y1_int = std::make_shared<ov::op::v0::Convert>(y1_bounded, ov::element::i32);
            
            // Calculate linear indices for the 4 corner points
            auto w_in_i32_bil = std::make_shared<ov::op::v0::Convert>(w_in, ov::element::i32);
            auto linear_idx_00 = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(y0_int, w_in_i32_bil), x0_int);
            auto linear_idx_01 = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(y0_int, w_in_i32_bil), x1_int);
            auto linear_idx_10 = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(y1_int, w_in_i32_bil), x0_int);
            auto linear_idx_11 = std::make_shared<ov::op::v1::Add>(
                std::make_shared<ov::op::v1::Multiply>(y1_int, w_in_i32_bil), x1_int);
            
            // Билинейная интерполяция без Gather: используем OneHot + Multiply + ReduceSum
            // 1. Привести input к плоскому виду: [N, C, H_in, W_in] -> [N, C, H_in * W_in]
            auto hw_dim_bil = std::make_shared<ov::op::v1::Multiply>(h_in, w_in);
            auto input_flat_shape_bil = std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{n_dim, c_dim, hw_dim_bil}, 0);
            auto input_flat_bil = std::make_shared<ov::op::v1::Reshape>(data, input_flat_shape_bil, false);
            
            // Helper function для создания значений в углах через OneHot
            auto sample_corner_values = [&](const std::shared_ptr<ov::Node>& linear_idx) -> std::shared_ptr<ov::Node> {
                
                // Use linear index directly
                std::shared_ptr<ov::Node> final_linear_idx = linear_idx;
                
                // Создаем OneHot маску для индексов: [N, H_out, W_out] -> [N, H_out, W_out, H_in*W_in]
                auto hw_dim_i32_bil = std::make_shared<ov::op::v0::Convert>(hw_dim_bil, ov::element::i32);
                auto hw_dim_scalar_bil = std::make_shared<ov::op::v0::Squeeze>(hw_dim_i32_bil);
                auto one_hot_mask = std::make_shared<ov::op::v1::OneHot>(
                    final_linear_idx,
                    hw_dim_scalar_bil,
                    ov::op::v0::Constant::create(element_type, {}, {1.0f}),
                    ov::op::v0::Constant::create(element_type, {}, {0.0f}),
                    -1);
                
                // Расширяем input_flat для broadcasting: [N, C, H_in*W_in] -> [N, C, 1, 1, H_in*W_in]
                auto input_expanded = std::make_shared<ov::op::v0::Unsqueeze>(input_flat_bil, 
                    ov::op::v0::Constant::create(ov::element::i32, {1}, {2}));
                input_expanded = std::make_shared<ov::op::v0::Unsqueeze>(input_expanded, 
                    ov::op::v0::Constant::create(ov::element::i32, {1}, {2}));
                
                // Расширяем one_hot_mask для broadcasting: [N, H_out, W_out, H_in*W_in] -> [N, 1, H_out, W_out, H_in*W_in]
                auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(one_hot_mask, 
                    ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
                
                // Multiply + ReduceSum для эмуляции gather
                auto multiplied = std::make_shared<ov::op::v1::Multiply>(input_expanded, mask_expanded);
                auto reduce_axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {4});
                std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v1::ReduceSum>(multiplied, reduce_axis, false);
                
                // Note: zeros padding will be applied to the final interpolated result, not here
                
                return result;
            };
            
            // Sample values at the 4 corner points
            std::shared_ptr<ov::Node> values_00 = sample_corner_values(linear_idx_00);
            std::shared_ptr<ov::Node> values_01 = sample_corner_values(linear_idx_01);
            std::shared_ptr<ov::Node> values_10 = sample_corner_values(linear_idx_10);
            std::shared_ptr<ov::Node> values_11 = sample_corner_values(linear_idx_11);
            
            // Compute bilinear interpolation weights
            auto w_00 = std::make_shared<ov::op::v1::Multiply>(x1_minus_x, y1_minus_y);
            auto w_01 = std::make_shared<ov::op::v1::Multiply>(x_minus_x0, y1_minus_y);
            auto w_10 = std::make_shared<ov::op::v1::Multiply>(x1_minus_x, y_minus_y0);
            auto w_11 = std::make_shared<ov::op::v1::Multiply>(x_minus_x0, y_minus_y0);
            
            // Add channel dimension to weights for broadcasting [N, H_out, W_out] -> [N, 1, H_out, W_out]
            auto unsqueeze_ax_bil = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
            auto w_00_exp = std::make_shared<ov::op::v0::Unsqueeze>(w_00, unsqueeze_ax_bil);
            auto w_01_exp = std::make_shared<ov::op::v0::Unsqueeze>(w_01, unsqueeze_ax_bil);
            auto w_10_exp = std::make_shared<ov::op::v0::Unsqueeze>(w_10, unsqueeze_ax_bil);
            auto w_11_exp = std::make_shared<ov::op::v0::Unsqueeze>(w_11, unsqueeze_ax_bil);
            
            // Perform weighted sum
            auto weighted_00 = std::make_shared<ov::op::v1::Multiply>(values_00, w_00_exp);
            auto weighted_01 = std::make_shared<ov::op::v1::Multiply>(values_01, w_01_exp);
            auto weighted_10 = std::make_shared<ov::op::v1::Multiply>(values_10, w_10_exp);
            auto weighted_11 = std::make_shared<ov::op::v1::Multiply>(values_11, w_11_exp);
            
            auto sum_0 = std::make_shared<ov::op::v1::Add>(weighted_00, weighted_01);
            auto sum_1 = std::make_shared<ov::op::v1::Add>(weighted_10, weighted_11);
            interpolated = std::make_shared<ov::op::v1::Add>(sum_0, sum_1);
            
            // Apply zeros padding mask based on original coordinates
            if (is_zeros) {
                auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                    std::make_shared<ov::op::v1::GreaterEqual>(x, zero),
                    std::make_shared<ov::op::v1::Less>(x, w_in_f));
                auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                    std::make_shared<ov::op::v1::GreaterEqual>(y, zero),
                    std::make_shared<ov::op::v1::Less>(y, h_in_f));
                auto coord_valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                auto coord_valid_f = std::make_shared<ov::op::v0::Convert>(coord_valid, element_type);
                
                // Add channel dimension: [N, H_out, W_out] -> [N, 1, H_out, W_out]
                auto unsqueeze_axis_1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
                auto coord_valid_exp = std::make_shared<ov::op::v0::Unsqueeze>(coord_valid_f, unsqueeze_axis_1);
                
                interpolated = std::make_shared<ov::op::v1::Multiply>(interpolated, coord_valid_exp);
            }
        } else if (is_bicubic) {
            // Бикубическая интерполяция без Gather: используем OneHot + Multiply + ReduceSum
            // Нужна сетка 4x4: от (x0-1, y0-1) до (x0+2, y0+2)
            
            // 1. Вычисляем дробные части dx = x - floor(x), dy = y - floor(y)
            auto dx = std::make_shared<ov::op::v1::Subtract>(x, x0);
            auto dy = std::make_shared<ov::op::v1::Subtract>(y, y0);
            
            // 2. Вычисляем координаты сетки 4x4
            auto x_minus_1 = std::make_shared<ov::op::v1::Subtract>(x0, const_1);
            auto x_plus_2 = std::make_shared<ov::op::v1::Add>(x0, const_2);
            auto y_minus_1 = std::make_shared<ov::op::v1::Subtract>(y0, const_1);
            auto y_plus_2 = std::make_shared<ov::op::v1::Add>(y0, const_2);
            
            std::vector<std::shared_ptr<ov::Node>> x_coords = {x_minus_1, x0, x1, x_plus_2};
            std::vector<std::shared_ptr<ov::Node>> y_coords = {y_minus_1, y0, y1, y_plus_2};
            
            // 3. Преобразуем входные данные: [N, C, H_in, W_in] -> [N, C, H_in*W_in]
            auto hw_dim_bic = std::make_shared<ov::op::v1::Multiply>(h_in, w_in);
            auto input_flat_shape_bic = std::make_shared<ov::op::v0::Concat>(
                std::vector<Output<Node>>{n_dim, c_dim, hw_dim_bic}, 0);
            auto input_flat_bic = std::make_shared<ov::op::v1::Reshape>(data, input_flat_shape_bic, false);
            
            // 4. Создаем вспомогательные константы для границ
            auto zero_i32 = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
            auto h_in_i32 = std::make_shared<ov::op::v0::Convert>(h_in, ov::element::i32);
            auto w_in_i32 = std::make_shared<ov::op::v0::Convert>(w_in, ov::element::i32);
            auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_i32, ov::op::v0::Constant::create(ov::element::i32, {}, {1}));
            auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_i32, ov::op::v0::Constant::create(ov::element::i32, {}, {1}));
            
            // 5. Создаем функцию для выборки значений с помощью OneHot
            auto sample_bicubic_values = [&](std::shared_ptr<ov::Node> x_coord, std::shared_ptr<ov::Node> y_coord) -> std::shared_ptr<ov::Node> {
                // Преобразуем координаты в int32 и применяем clamp
                auto x_int = std::make_shared<ov::op::v0::Convert>(x_coord, ov::element::i32);
                auto y_int = std::make_shared<ov::op::v0::Convert>(y_coord, ov::element::i32);
                
                auto x_clamped = std::make_shared<ov::op::v1::Maximum>(zero_i32, 
                    std::make_shared<ov::op::v1::Minimum>(x_int, w_minus_1));
                auto y_clamped = std::make_shared<ov::op::v1::Maximum>(zero_i32,
                    std::make_shared<ov::op::v1::Minimum>(y_int, h_minus_1));
                
                // Вычисляем линейный индекс: y*W + x
                auto linear_idx = std::make_shared<ov::op::v1::Add>(
                    std::make_shared<ov::op::v1::Multiply>(y_clamped, w_in_i32), x_clamped);
                
                // Создаем OneHot маску: [N, H_out, W_out] -> [N, H_out, W_out, H_in*W_in]
                auto hw_dim_i32_bic = std::make_shared<ov::op::v0::Convert>(hw_dim_bic, ov::element::i32);
                auto hw_dim_scalar_bic = std::make_shared<ov::op::v0::Squeeze>(hw_dim_i32_bic);
                auto one_hot_mask = std::make_shared<ov::op::v1::OneHot>(
                    linear_idx,
                    hw_dim_scalar_bic,
                    ov::op::v0::Constant::create(element_type, {}, {1.0F}),
                    ov::op::v0::Constant::create(element_type, {}, {0.0F}),
                    -1);
                
                // Расширяем размерности для broadcasting
                // input_flat_bic: [N, C, H_in*W_in] -> [N, C, 1, 1, H_in*W_in]
                auto input_expanded = std::make_shared<ov::op::v0::Unsqueeze>(input_flat_bic,
                    ov::op::v0::Constant::create(ov::element::i32, {1}, {2}));
                input_expanded = std::make_shared<ov::op::v0::Unsqueeze>(input_expanded,
                    ov::op::v0::Constant::create(ov::element::i32, {1}, {2}));
                
                // one_hot_mask: [N, H_out, W_out, H_in*W_in] -> [N, 1, H_out, W_out, H_in*W_in]
                auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(one_hot_mask,
                    ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
                
                // Эмулируем gather: Multiply + ReduceSum
                auto multiplied = std::make_shared<ov::op::v1::Multiply>(input_expanded, mask_expanded);
                auto reduce_axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {4});
                std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v1::ReduceSum>(multiplied, reduce_axis, false);
                
                // Для zeros padding: умножаем на валидность координат
                if (is_zeros) {
                    auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                        std::make_shared<ov::op::v1::GreaterEqual>(x_coord, zero),
                        std::make_shared<ov::op::v1::Less>(x_coord, w_in_f));
                    auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                        std::make_shared<ov::op::v1::GreaterEqual>(y_coord, zero),
                        std::make_shared<ov::op::v1::Less>(y_coord, h_in_f));
                    auto coord_valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                    auto coord_valid_f = std::make_shared<ov::op::v0::Convert>(coord_valid, element_type);
                    
                    // Добавляем channel dimension: [N, H_out, W_out] -> [N, 1, H_out, W_out]
                    auto coord_valid_exp = std::make_shared<ov::op::v0::Unsqueeze>(coord_valid_f,
                        ov::op::v0::Constant::create(ov::element::i32, {1}, {1}));
                    
                    result = std::make_shared<ov::op::v1::Multiply>(result, coord_valid_exp);
                }
                
                return result;
            };
            
            // 6. Получаем значения для всех 16 точек сетки 4x4
            std::vector<std::shared_ptr<ov::Node>> values_4x4;
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    values_4x4.push_back(sample_bicubic_values(x_coords[i], y_coords[j]));
                }
            }
            
            // 7. Вычисляем кубические коэффициенты
            // A = -0.75 (стандартный параметр для бикубической интерполяции)
            auto A = ov::op::v0::Constant::create(element_type, {}, {-0.75F});
            auto const_2 = ov::op::v0::Constant::create(element_type, {}, {2.0F});
            auto const_3 = ov::op::v0::Constant::create(element_type, {}, {3.0F});
            auto const_4 = ov::op::v0::Constant::create(element_type, {}, {4.0F});
            auto const_5 = ov::op::v0::Constant::create(element_type, {}, {5.0F});
            auto const_8 = ov::op::v0::Constant::create(element_type, {}, {8.0F});
            
            // 8. Вычисляем кубические коэффициенты для X направления
            // cx[0] = ((A * (dx + 1) - 5 * A) * (dx + 1) + 8 * A) * (dx + 1) - 4 * A
            auto dx_plus_1 = std::make_shared<ov::op::v1::Add>(dx, const_1);
            auto five_A = std::make_shared<ov::op::v1::Multiply>(const_5, A);
            auto eight_A = std::make_shared<ov::op::v1::Multiply>(const_8, A);
            auto four_A = std::make_shared<ov::op::v1::Multiply>(const_4, A);
            auto A_plus_2_bic = std::make_shared<ov::op::v1::Add>(A, const_2);
            auto A_plus_3 = std::make_shared<ov::op::v1::Add>(A, const_3);
            
            auto cx0_t1 = std::make_shared<ov::op::v1::Multiply>(A, dx_plus_1);
            auto cx0_t2 = std::make_shared<ov::op::v1::Subtract>(cx0_t1, five_A);
            auto cx0_t3 = std::make_shared<ov::op::v1::Multiply>(cx0_t2, dx_plus_1);
            auto cx0_t4 = std::make_shared<ov::op::v1::Add>(cx0_t3, eight_A);
            auto cx0_t5 = std::make_shared<ov::op::v1::Multiply>(cx0_t4, dx_plus_1);
            auto cx0 = std::make_shared<ov::op::v1::Subtract>(cx0_t5, four_A);
            
            // cx[1] = ((A + 2) * dx - (A + 3)) * dx * dx + 1
            auto cx1_t1 = std::make_shared<ov::op::v1::Multiply>(A_plus_2_bic, dx);
            auto cx1_t2 = std::make_shared<ov::op::v1::Subtract>(cx1_t1, A_plus_3);
            auto dx_squared = std::make_shared<ov::op::v1::Multiply>(dx, dx);
            auto cx1_t3 = std::make_shared<ov::op::v1::Multiply>(cx1_t2, dx_squared);
            auto cx1 = std::make_shared<ov::op::v1::Add>(cx1_t3, const_1);
            
            // cx[2] = ((A + 2) * (1 - dx) - (A + 3)) * (1 - dx) * (1 - dx) + 1
            auto one_minus_dx = std::make_shared<ov::op::v1::Subtract>(const_1, dx);
            auto cx2_t1 = std::make_shared<ov::op::v1::Multiply>(A_plus_2_bic, one_minus_dx);
            auto cx2_t2 = std::make_shared<ov::op::v1::Subtract>(cx2_t1, A_plus_3);
            auto one_minus_dx_squared = std::make_shared<ov::op::v1::Multiply>(one_minus_dx, one_minus_dx);
            auto cx2_t3 = std::make_shared<ov::op::v1::Multiply>(cx2_t2, one_minus_dx_squared);
            auto cx2 = std::make_shared<ov::op::v1::Add>(cx2_t3, const_1);
            
            // cx[3] = ((A * (2 - dx) - 5 * A) * (2 - dx) + 8 * A) * (2 - dx) - 4 * A
            auto two_minus_dx = std::make_shared<ov::op::v1::Subtract>(const_2, dx);
            auto cx3_t1 = std::make_shared<ov::op::v1::Multiply>(A, two_minus_dx);
            auto cx3_t2 = std::make_shared<ov::op::v1::Subtract>(cx3_t1, five_A);
            auto cx3_t3 = std::make_shared<ov::op::v1::Multiply>(cx3_t2, two_minus_dx);
            auto cx3_t4 = std::make_shared<ov::op::v1::Add>(cx3_t3, eight_A);
            auto cx3_t5 = std::make_shared<ov::op::v1::Multiply>(cx3_t4, two_minus_dx);
            auto cx3 = std::make_shared<ov::op::v1::Subtract>(cx3_t5, four_A);
            
            // 9. Аналогично для Y направления
            auto dy_plus_1 = std::make_shared<ov::op::v1::Add>(dy, const_1);
            auto cy0_t1 = std::make_shared<ov::op::v1::Multiply>(A, dy_plus_1);
            auto cy0_t2 = std::make_shared<ov::op::v1::Subtract>(cy0_t1, five_A);
            auto cy0_t3 = std::make_shared<ov::op::v1::Multiply>(cy0_t2, dy_plus_1);
            auto cy0_t4 = std::make_shared<ov::op::v1::Add>(cy0_t3, eight_A);
            auto cy0_t5 = std::make_shared<ov::op::v1::Multiply>(cy0_t4, dy_plus_1);
            auto cy0 = std::make_shared<ov::op::v1::Subtract>(cy0_t5, four_A);
            
            auto cy1_t1 = std::make_shared<ov::op::v1::Multiply>(A_plus_2_bic, dy);
            auto cy1_t2 = std::make_shared<ov::op::v1::Subtract>(cy1_t1, A_plus_3);
            auto dy_squared = std::make_shared<ov::op::v1::Multiply>(dy, dy);
            auto cy1_t3 = std::make_shared<ov::op::v1::Multiply>(cy1_t2, dy_squared);
            auto cy1 = std::make_shared<ov::op::v1::Add>(cy1_t3, const_1);
            
            auto one_minus_dy = std::make_shared<ov::op::v1::Subtract>(const_1, dy);
            auto cy2_t1 = std::make_shared<ov::op::v1::Multiply>(A_plus_2_bic, one_minus_dy);
            auto cy2_t2 = std::make_shared<ov::op::v1::Subtract>(cy2_t1, A_plus_3);
            auto one_minus_dy_squared = std::make_shared<ov::op::v1::Multiply>(one_minus_dy, one_minus_dy);
            auto cy2_t3 = std::make_shared<ov::op::v1::Multiply>(cy2_t2, one_minus_dy_squared);
            auto cy2 = std::make_shared<ov::op::v1::Add>(cy2_t3, const_1);
            
            auto two_minus_dy = std::make_shared<ov::op::v1::Subtract>(const_2, dy);
            auto cy3_t1 = std::make_shared<ov::op::v1::Multiply>(A, two_minus_dy);
            auto cy3_t2 = std::make_shared<ov::op::v1::Subtract>(cy3_t1, five_A);
            auto cy3_t3 = std::make_shared<ov::op::v1::Multiply>(cy3_t2, two_minus_dy);
            auto cy3_t4 = std::make_shared<ov::op::v1::Add>(cy3_t3, eight_A);
            auto cy3_t5 = std::make_shared<ov::op::v1::Multiply>(cy3_t4, two_minus_dy);
            auto cy3 = std::make_shared<ov::op::v1::Subtract>(cy3_t5, four_A);
            
            // 10. Добавляем channel dimension к коэффициентам для broadcasting
            auto unsqueeze_ax = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
            auto cx0_exp = std::make_shared<ov::op::v0::Unsqueeze>(cx0, unsqueeze_ax);
            auto cx1_exp = std::make_shared<ov::op::v0::Unsqueeze>(cx1, unsqueeze_ax);
            auto cx2_exp = std::make_shared<ov::op::v0::Unsqueeze>(cx2, unsqueeze_ax);
            auto cx3_exp = std::make_shared<ov::op::v0::Unsqueeze>(cx3, unsqueeze_ax);
            
            auto cy0_exp = std::make_shared<ov::op::v0::Unsqueeze>(cy0, unsqueeze_ax);
            auto cy1_exp = std::make_shared<ov::op::v0::Unsqueeze>(cy1, unsqueeze_ax);
            auto cy2_exp = std::make_shared<ov::op::v0::Unsqueeze>(cy2, unsqueeze_ax);
            auto cy3_exp = std::make_shared<ov::op::v0::Unsqueeze>(cy3, unsqueeze_ax);
            
            // 11. Сначала интерполируем по X направлению (4 строки)
            std::vector<std::shared_ptr<ov::Node>> p_rows;
            for (int j = 0; j < 4; ++j) {
                auto p0 = std::make_shared<ov::op::v1::Multiply>(values_4x4[(j*4) + 0], cx0_exp);
                auto p1 = std::make_shared<ov::op::v1::Multiply>(values_4x4[(j*4) + 1], cx1_exp);
                auto p2 = std::make_shared<ov::op::v1::Multiply>(values_4x4[(j*4) + 2], cx2_exp);
                auto p3 = std::make_shared<ov::op::v1::Multiply>(values_4x4[(j*4) + 3], cx3_exp);
                
                auto sum_01 = std::make_shared<ov::op::v1::Add>(p0, p1);
                auto sum_23 = std::make_shared<ov::op::v1::Add>(p2, p3);
                auto row_result = std::make_shared<ov::op::v1::Add>(sum_01, sum_23);
                p_rows.push_back(row_result);
            }
            
            // 12. Затем интерполируем по Y направлению
            auto final_p0 = std::make_shared<ov::op::v1::Multiply>(p_rows[0], cy0_exp);
            auto final_p1 = std::make_shared<ov::op::v1::Multiply>(p_rows[1], cy1_exp);
            auto final_p2 = std::make_shared<ov::op::v1::Multiply>(p_rows[2], cy2_exp);
            auto final_p3 = std::make_shared<ov::op::v1::Multiply>(p_rows[3], cy3_exp);
            
            auto final_sum01 = std::make_shared<ov::op::v1::Add>(final_p0, final_p1);
            auto final_sum23 = std::make_shared<ov::op::v1::Add>(final_p2, final_p3);
            interpolated = std::make_shared<ov::op::v1::Add>(final_sum01, final_sum23);
            
            // Apply zeros padding mask based on original coordinates
            if (is_zeros) {
                auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                    std::make_shared<ov::op::v1::GreaterEqual>(x, zero),
                    std::make_shared<ov::op::v1::Less>(x, w_in_f));
                auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                    std::make_shared<ov::op::v1::GreaterEqual>(y, zero),
                    std::make_shared<ov::op::v1::Less>(y, h_in_f));
                auto coord_valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                auto coord_valid_f = std::make_shared<ov::op::v0::Convert>(coord_valid, element_type);
                
                // Add channel dimension: [N, H_out, W_out] -> [N, 1, H_out, W_out]
                auto unsqueeze_axis_1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
                auto coord_valid_exp = std::make_shared<ov::op::v0::Unsqueeze>(coord_valid_f, unsqueeze_axis_1);
                
                interpolated = std::make_shared<ov::op::v1::Multiply>(interpolated, coord_valid_exp);
            }
        }
        
        // Result is already in correct shape [N, C, H_out, W_out]
        auto result = interpolated;
        
        // Copy runtime info and replace node
        result->set_friendly_name(grid_sample->get_friendly_name());
        std::vector<std::shared_ptr<ov::Node>> rt_info_nodes = {x, y, result};
        if (is_bicubic) {  // TEMPORARY: Disable for bilinear to debug
            rt_info_nodes.insert(rt_info_nodes.end(), {x0, x1, y0, y1});
        }
        if (is_nearest) {
            rt_info_nodes.insert(rt_info_nodes.end(), {x_idx, y_idx});
        }
        ov::copy_runtime_info(grid_sample, rt_info_nodes);
        ov::replace_node(grid_sample, result);
        
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(grid_sample_pattern, "GridSampleDecomposition");
    register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov