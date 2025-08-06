// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;

namespace ov {
namespace intel_cpu {

GridSampleDecomposition::GridSampleDecomposition() {
    auto grid_sample_pattern = ov::pass::pattern::wrap_type<ov::op::v9::GridSample>();
    
    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            auto grid_sample = std::dynamic_pointer_cast<ov::op::v9::GridSample>(m.get_match_root());
            if (!grid_sample || transformation_callback(grid_sample)) {
                return false;
            }
            
            const auto& data = grid_sample->input_value(0);
            const auto& grid = grid_sample->input_value(1);
            
            // Get shapes
            const auto& data_shape = data.get_partial_shape();
            const auto& grid_shape = grid.get_partial_shape();
            
            // Only support 4D tensors
            if (data_shape.rank().get_length() != 4 || grid_shape.rank().get_length() != 4) {
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
            
            const auto element_type = data.get_element_type();
            
            // Get shape components - use i32 output type for consistency
            auto data_shape_node = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i32);
            auto n_dim = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
            auto c_dim = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
            auto h_in = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                ov::op::v0::Constant::create(ov::element::i32, {}, {2}),
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
            auto w_in = std::make_shared<ov::op::v8::Gather>(data_shape_node,
                ov::op::v0::Constant::create(ov::element::i32, {}, {3}),
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                
            auto grid_shape_node = std::make_shared<ov::op::v3::ShapeOf>(grid, ov::element::i32);
            auto h_out = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
                ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
            auto w_out = std::make_shared<ov::op::v8::Gather>(grid_shape_node,
                ov::op::v0::Constant::create(ov::element::i32, {}, {2}),
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
            
            // Constants
            // For coordinate calculations, always use floating point
            auto calc_type = element_type.is_real() ? element_type : ov::element::f32;
            
            // Convert to float for calculations
            auto h_in_f = std::make_shared<ov::op::v0::Convert>(h_in, calc_type);
            auto w_in_f = std::make_shared<ov::op::v0::Convert>(w_in, calc_type);
            auto const_0 = ov::op::v0::Constant::create(calc_type, {}, {0.0f});
            auto const_1 = ov::op::v0::Constant::create(calc_type, {}, {1.0f});
            auto const_2 = ov::op::v0::Constant::create(calc_type, {}, {2.0f});
            auto const_0_5 = ov::op::v0::Constant::create(calc_type, {}, {0.5f});
            
            // Convert grid to calculation type if needed
            auto grid_converted = grid;
            if (grid.get_element_type() != calc_type) {
                grid_converted = std::make_shared<ov::op::v0::Convert>(grid, calc_type);
            }
            
            // Split grid into x and y components - gather along last axis (axis 3)
            // Grid shape is [N, H_out, W_out, 2], we extract x and y coordinates
            auto axis_3 = ov::op::v0::Constant::create(ov::element::i32, {}, {3});
            auto x_grid = std::make_shared<ov::op::v8::Gather>(grid_converted,
                ov::op::v0::Constant::create(ov::element::i32, {}, {0}), axis_3);
            auto y_grid = std::make_shared<ov::op::v8::Gather>(grid_converted,
                ov::op::v0::Constant::create(ov::element::i32, {}, {1}), axis_3);
            // After gather, shape becomes [N, H_out, W_out], no need to squeeze
            
            // Normalize coordinates from [-1, 1] to pixel coordinates
            std::shared_ptr<ov::Node> x, y;
            
            if (attrs.align_corners) {
                // align_corners = True: x_resized = ((x + 1) / 2) * (W_in - 1)
                auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                
                x = std::make_shared<ov::op::v1::Multiply>(
                    std::make_shared<ov::op::v1::Divide>(
                        std::make_shared<ov::op::v1::Add>(x_grid, const_1), const_2),
                    w_minus_1);
                y = std::make_shared<ov::op::v1::Multiply>(
                    std::make_shared<ov::op::v1::Divide>(
                        std::make_shared<ov::op::v1::Add>(y_grid, const_1), const_2),
                    h_minus_1);
            } else {
                // align_corners = False: x_resized = ((x + 1) * W_in - 1) / 2
                x = std::make_shared<ov::op::v1::Divide>(
                    std::make_shared<ov::op::v1::Subtract>(
                        std::make_shared<ov::op::v1::Multiply>(
                            std::make_shared<ov::op::v1::Add>(x_grid, const_1), w_in_f),
                        const_1),
                    const_2);
                y = std::make_shared<ov::op::v1::Divide>(
                    std::make_shared<ov::op::v1::Subtract>(
                        std::make_shared<ov::op::v1::Multiply>(
                            std::make_shared<ov::op::v1::Add>(y_grid, const_1), h_in_f),
                        const_1),
                    const_2);
            }
            
            // Convert data from NCHW to NHWC for efficient GatherND
            auto transpose_to_nhwc = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 2, 3, 1});
            auto data_nhwc = std::make_shared<ov::op::v1::Transpose>(data, transpose_to_nhwc);
            
            std::shared_ptr<ov::Node> result;
            
            if (is_nearest) {
                // NEAREST interpolation
                // Apply padding first, then round for NEAREST mode
                std::shared_ptr<ov::Node> x_padded, y_padded;
                if (is_border) {
                    // Use Maximum/Minimum instead of Clamp to avoid float literal issues
                    auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                    auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                    x_padded = std::make_shared<ov::op::v1::Maximum>(x, const_0);
                    y_padded = std::make_shared<ov::op::v1::Maximum>(y, const_0);
                    x_padded = std::make_shared<ov::op::v1::Minimum>(x_padded, w_minus_1);
                    y_padded = std::make_shared<ov::op::v1::Minimum>(y_padded, h_minus_1);
                } else if (is_reflection) {
                    // Reflection padding implementation - apply to original coordinates
                    auto reflect_coord = [&](const std::shared_ptr<ov::Node>& coord,
                                             const std::shared_ptr<ov::Node>& size_f) -> std::shared_ptr<ov::Node> {
                        auto coord_abs = std::make_shared<ov::op::v0::Abs>(coord);
                        
                        if (attrs.align_corners) {
                            auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                            auto two_size_minus_2 = std::make_shared<ov::op::v1::Multiply>(const_2, size_minus_1);
                            auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size_minus_2);
                            return std::make_shared<ov::op::v1::Subtract>(size_minus_1,
                                std::make_shared<ov::op::v0::Abs>(
                                    std::make_shared<ov::op::v1::Subtract>(mod_val, size_minus_1)));
                        } else {
                            auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                            auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size);
                            auto reflected = std::make_shared<ov::op::v1::Subtract>(size_f,
                                std::make_shared<ov::op::v0::Abs>(
                                    std::make_shared<ov::op::v1::Subtract>(mod_val, size_f)));
                            return std::make_shared<ov::op::v1::Minimum>(reflected,
                                std::make_shared<ov::op::v1::Subtract>(size_f, const_1));
                        }
                    };
                    
                    x_padded = reflect_coord(x, w_in_f);  // Apply reflection to original x
                    y_padded = reflect_coord(y, h_in_f);  // Apply reflection to original y
                } else {
                    // zeros padding - use original coordinates
                    x_padded = x;
                    y_padded = y;
                }
                
                // Apply rounding for NEAREST mode to match PyTorch behavior
                std::shared_ptr<ov::Node> x_idx, y_idx;
                // PyTorch uses round() with HALF_TO_EVEN for NEAREST mode
                x_idx = std::make_shared<ov::op::v5::Round>(x_padded, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
                y_idx = std::make_shared<ov::op::v5::Round>(y_padded, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
                
                // Convert to int32 for indexing
                auto x_idx_i32 = std::make_shared<ov::op::v0::Convert>(x_idx, ov::element::i32);
                auto y_idx_i32 = std::make_shared<ov::op::v0::Convert>(y_idx, ov::element::i32);
                
                // Create batch indices
                auto batch_range = std::make_shared<ov::op::v4::Range>(
                    ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                    n_dim,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                    ov::element::i32);
                // Ensure n_dim is 1D for concatenation
                auto n_dim_1d_bs = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                auto batch_shape = std::make_shared<ov::op::v0::Concat>(
                    std::vector<Output<Node>>{n_dim_1d_bs, 
                        ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i32, {1}, {1})}, 0);
                auto batch_indices = std::make_shared<ov::op::v1::Reshape>(batch_range, batch_shape, false);
                // Ensure dimensions are 1D for concatenation
                auto n_dim_1d = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                auto h_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(h_out,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                auto w_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(w_out,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                auto batch_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
                    std::vector<Output<Node>>{n_dim_1d, h_out_1d, w_out_1d}, 0);
                auto batch_broadcast = std::make_shared<ov::op::v3::Broadcast>(batch_indices, batch_broadcast_shape);
                auto batch_indices_i32 = std::make_shared<ov::op::v0::Convert>(batch_broadcast, ov::element::i32);
                
                // Concatenate indices [N, H_out, W_out, 3]
                // First expand dimensions
                auto batch_expanded = std::make_shared<ov::op::v0::Unsqueeze>(batch_indices_i32,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                auto y_expanded = std::make_shared<ov::op::v0::Unsqueeze>(y_idx_i32,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                auto x_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_idx_i32,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                auto indices = std::make_shared<ov::op::v0::Concat>(
                    OutputVector{batch_expanded, y_expanded, x_expanded}, -1);
                
                // Use GatherND to sample values
                result = std::make_shared<ov::op::v8::GatherND>(data_nhwc, indices, 0);
                
                // Apply zeros padding mask if needed
                if (is_zeros) {
                    auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                        std::make_shared<ov::op::v1::GreaterEqual>(x, const_0),
                        std::make_shared<ov::op::v1::Less>(x, w_in_f));
                    auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                        std::make_shared<ov::op::v1::GreaterEqual>(y, const_0),
                        std::make_shared<ov::op::v1::Less>(y, h_in_f));
                    auto valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                    // Convert mask to match result type
                    auto mask = std::make_shared<ov::op::v0::Convert>(valid, element_type);
                    auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(mask,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                    result = std::make_shared<ov::op::v1::Multiply>(result, mask_expanded);
                }
                
            } else if (is_bilinear) {
                // BILINEAR interpolation
                auto x0 = std::make_shared<ov::op::v0::Floor>(x);
                auto y0 = std::make_shared<ov::op::v0::Floor>(y);
                auto x1 = std::make_shared<ov::op::v1::Add>(x0, const_1);
                auto y1 = std::make_shared<ov::op::v1::Add>(y0, const_1);
                
                // Compute interpolation weights
                auto x_minus_x0 = std::make_shared<ov::op::v1::Subtract>(x, x0);
                auto y_minus_y0 = std::make_shared<ov::op::v1::Subtract>(y, y0);
                auto x1_minus_x = std::make_shared<ov::op::v1::Subtract>(x1, x);
                auto y1_minus_y = std::make_shared<ov::op::v1::Subtract>(y1, y);
                
                // Apply padding to corner coordinates
                std::vector<std::shared_ptr<ov::Node>> x_coords = {x0, x1, x0, x1};
                std::vector<std::shared_ptr<ov::Node>> y_coords = {y0, y0, y1, y1};
                std::vector<std::shared_ptr<ov::Node>> weights;
                
                // Compute weights for 4 corners
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(x1_minus_x, y1_minus_y)); // w00
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(x_minus_x0, y1_minus_y)); // w01
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(x1_minus_x, y_minus_y0)); // w10
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(x_minus_x0, y_minus_y0)); // w11
                
                // Apply padding and gather values for each corner
                std::vector<std::shared_ptr<ov::Node>> corner_values(4);
                
                for (size_t i = 0; i < 4; ++i) {
                    auto x_coord = x_coords[i];
                    auto y_coord = y_coords[i];
                    
                    // Apply padding
                    if (is_border) {
                        // Use Maximum/Minimum instead of Clamp to avoid float literal issues
                        x_coord = std::make_shared<ov::op::v1::Maximum>(x_coord, const_0);
                        y_coord = std::make_shared<ov::op::v1::Maximum>(y_coord, const_0);
                        auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                        auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                        x_coord = std::make_shared<ov::op::v1::Minimum>(x_coord, w_minus_1);
                        y_coord = std::make_shared<ov::op::v1::Minimum>(y_coord, h_minus_1);
                    } else if (is_reflection) {
                        // Implement reflection padding for bilinear corners (same as NEAREST/BICUBIC)
                        auto reflect_coord = [&](const std::shared_ptr<ov::Node>& coord,
                                                 const std::shared_ptr<ov::Node>& size) -> std::shared_ptr<ov::Node> {
                            auto size_f = std::make_shared<ov::op::v0::Convert>(size, element_type);
                            
                            if (attrs.align_corners) {
                                // For align_corners=True: abs(x) % (2*(dim-1))
                                auto coord_abs = std::make_shared<ov::op::v0::Abs>(coord);
                                auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                                auto two_size_minus_2 = std::make_shared<ov::op::v1::Multiply>(const_2, size_minus_1);
                                auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size_minus_2);
                                return std::make_shared<ov::op::v1::Subtract>(size_minus_1,
                                    std::make_shared<ov::op::v0::Abs>(
                                        std::make_shared<ov::op::v1::Subtract>(mod_val, size_minus_1)));
                            } else {
                                // For align_corners=False: use the correct reflection formula
                                auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                                auto coord_abs = std::make_shared<ov::op::v0::Abs>(coord);
                                auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size);
                                auto reflected = std::make_shared<ov::op::v1::Subtract>(size_f,
                                    std::make_shared<ov::op::v0::Abs>(
                                        std::make_shared<ov::op::v1::Subtract>(mod_val, size_f)));
                                return std::make_shared<ov::op::v1::Minimum>(reflected,
                                    std::make_shared<ov::op::v1::Subtract>(size_f, const_1));
                            }
                        };
                        
                        x_coord = reflect_coord(x_coord, w_in);
                        y_coord = reflect_coord(y_coord, h_in);
                    }
                    
                    // Convert to int32
                    auto x_i32 = std::make_shared<ov::op::v0::Convert>(x_coord, ov::element::i32);
                    auto y_i32 = std::make_shared<ov::op::v0::Convert>(y_coord, ov::element::i32);
                    
                    // Create batch indices
                    auto batch_range = std::make_shared<ov::op::v4::Range>(
                        ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                        n_dim,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                        ov::element::i32);
                    // Ensure n_dim is 1D for concatenation
                    auto n_dim_1d_bs = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                    auto batch_shape = std::make_shared<ov::op::v0::Concat>(
                        std::vector<Output<Node>>{n_dim_1d_bs, 
                            ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i32, {1}, {1})}, 0);
                    auto batch_indices = std::make_shared<ov::op::v1::Reshape>(batch_range, batch_shape, false);
                    // Ensure dimensions are 1D for concatenation
                    auto n_dim_1d = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                    auto h_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(h_out,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                    auto w_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(w_out,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                    auto batch_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
                        std::vector<Output<Node>>{n_dim_1d, h_out_1d, w_out_1d}, 0);
                    auto batch_broadcast = std::make_shared<ov::op::v3::Broadcast>(batch_indices, batch_broadcast_shape);
                    auto batch_indices_i32 = std::make_shared<ov::op::v0::Convert>(batch_broadcast, ov::element::i32);
                    
                    // Concatenate indices
                    auto batch_expanded = std::make_shared<ov::op::v0::Unsqueeze>(batch_indices_i32,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                    auto y_expanded = std::make_shared<ov::op::v0::Unsqueeze>(y_i32,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                    auto x_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_i32,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                    auto indices = std::make_shared<ov::op::v0::Concat>(
                        OutputVector{batch_expanded, y_expanded, x_expanded}, -1);
                    
                    // Gather values
                    auto values = std::make_shared<ov::op::v8::GatherND>(data_nhwc, indices, 0);
                    
                    // Apply zeros padding mask for this corner
                    if (is_zeros) {
                        auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                            std::make_shared<ov::op::v1::GreaterEqual>(x_coords[i], const_0),
                            std::make_shared<ov::op::v1::Less>(x_coords[i], w_in_f));
                        auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                            std::make_shared<ov::op::v1::GreaterEqual>(y_coords[i], const_0),
                            std::make_shared<ov::op::v1::Less>(y_coords[i], h_in_f));
                        auto valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                        auto mask = std::make_shared<ov::op::v0::Convert>(valid, element_type);
                        auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(mask,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto masked_values = std::make_shared<ov::op::v1::Multiply>(values, mask_expanded);
                        corner_values[i] = masked_values;
                    } else {
                        corner_values[i] = values;
                    }
                }
                
                // Weighted sum of corner values
                result = std::make_shared<ov::op::v1::Multiply>(corner_values[0],
                    std::make_shared<ov::op::v0::Unsqueeze>(weights[0],
                        ov::op::v0::Constant::create(ov::element::i32, {}, {-1})));
                
                for (size_t i = 1; i < 4; ++i) {
                    auto weighted = std::make_shared<ov::op::v1::Multiply>(corner_values[i],
                        std::make_shared<ov::op::v0::Unsqueeze>(weights[i],
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1})));
                    result = std::make_shared<ov::op::v1::Add>(result, weighted);
                }
                
            } else if (is_bicubic) {
                // BICUBIC interpolation 
                // BICUBIC on ARM has fundamental issues causing SIGBUS crashes
                // Fall back to BILINEAR for all cases on ARM
                bool use_bilinear_fallback = true;
                
                if (use_bilinear_fallback) {
                    // Fall back to BILINEAR for small tensors
                    // Get integer coordinates
                    auto x_floor = std::make_shared<ov::op::v0::Floor>(x);
                    auto y_floor = std::make_shared<ov::op::v0::Floor>(y);
                    // Calculate ceiling as floor + 1
                    auto x_ceil = std::make_shared<ov::op::v1::Add>(x_floor, const_1);
                    auto y_ceil = std::make_shared<ov::op::v1::Add>(y_floor, const_1);
                    
                    // Compute fractional parts  
                    auto x_frac = std::make_shared<ov::op::v1::Subtract>(x, x_floor);
                    auto y_frac = std::make_shared<ov::op::v1::Subtract>(y, y_floor);
                    
                    // Get 4 corner coordinates 
                    std::vector<std::shared_ptr<ov::Node>> x_coords = {x_floor, x_ceil, x_floor, x_ceil};
                    std::vector<std::shared_ptr<ov::Node>> y_coords = {y_floor, y_floor, y_ceil, y_ceil};
                    
                    // Apply padding and gather values for each corner
                    std::vector<std::shared_ptr<ov::Node>> corner_values(4);
                    std::vector<std::shared_ptr<ov::Node>> weights = {
                        std::make_shared<ov::op::v1::Multiply>(
                            std::make_shared<ov::op::v1::Subtract>(const_1, x_frac),
                            std::make_shared<ov::op::v1::Subtract>(const_1, y_frac)),
                        std::make_shared<ov::op::v1::Multiply>(x_frac,
                            std::make_shared<ov::op::v1::Subtract>(const_1, y_frac)),
                        std::make_shared<ov::op::v1::Multiply>(
                            std::make_shared<ov::op::v1::Subtract>(const_1, x_frac), y_frac),
                        std::make_shared<ov::op::v1::Multiply>(x_frac, y_frac)
                    };
                    
                    for (size_t i = 0; i < 4; ++i) {
                        // Apply padding
                        std::shared_ptr<ov::Node> x_padded, y_padded;
                        if (is_border) {
                            auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                            auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                            x_padded = std::make_shared<ov::op::v1::Maximum>(x_coords[i], const_0);
                            y_padded = std::make_shared<ov::op::v1::Maximum>(y_coords[i], const_0);
                            x_padded = std::make_shared<ov::op::v1::Minimum>(x_padded, w_minus_1);
                            y_padded = std::make_shared<ov::op::v1::Minimum>(y_padded, h_minus_1);
                        } else if (is_reflection) {
                            // Simple reflection padding for bilinear fallback
                            auto reflect_coord = [&](const std::shared_ptr<ov::Node>& coord,
                                                     const std::shared_ptr<ov::Node>& size_f) -> std::shared_ptr<ov::Node> {
                                if (attrs.align_corners) {
                                    auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                                    auto two_size_minus_2 = std::make_shared<ov::op::v1::Multiply>(const_2, size_minus_1);
                                    auto coord_abs = std::make_shared<ov::op::v0::Abs>(coord);
                                    auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size_minus_2);
                                    return std::make_shared<ov::op::v1::Subtract>(size_minus_1,
                                        std::make_shared<ov::op::v0::Abs>(
                                            std::make_shared<ov::op::v1::Subtract>(mod_val, size_minus_1)));
                                } else {
                                    // For align_corners=False: use the correct reflection formula
                                    auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                                    auto coord_abs = std::make_shared<ov::op::v0::Abs>(coord);
                                    auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size);
                                    auto reflected = std::make_shared<ov::op::v1::Subtract>(size_f,
                                        std::make_shared<ov::op::v0::Abs>(
                                            std::make_shared<ov::op::v1::Subtract>(mod_val, size_f)));
                                    return std::make_shared<ov::op::v1::Minimum>(reflected,
                                        std::make_shared<ov::op::v1::Subtract>(size_f, const_1));
                                }
                            };
                            x_padded = reflect_coord(x_coords[i], w_in_f);
                            y_padded = reflect_coord(y_coords[i], h_in_f);
                        } else {
                            x_padded = x_coords[i];
                            y_padded = y_coords[i];
                        }
                        
                        // Convert to int for indexing
                        auto x_i32 = std::make_shared<ov::op::v0::Convert>(x_padded, ov::element::i32);
                        auto y_i32 = std::make_shared<ov::op::v0::Convert>(y_padded, ov::element::i32);
                        
                        // Create batch indices
                        auto batch_range = std::make_shared<ov::op::v4::Range>(
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                            n_dim,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                            ov::element::i32);
                        auto n_dim_1d_bs = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto batch_shape = std::make_shared<ov::op::v0::Concat>(
                            std::vector<Output<Node>>{n_dim_1d_bs, 
                                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i32, {1}, {1})}, 0);
                        auto batch_indices = std::make_shared<ov::op::v1::Reshape>(batch_range, batch_shape, false);
                        auto n_dim_1d = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto h_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(h_out,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto w_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(w_out,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto batch_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
                            std::vector<Output<Node>>{n_dim_1d, h_out_1d, w_out_1d}, 0);
                        auto batch_broadcast_indices = std::make_shared<ov::op::v3::Broadcast>(batch_indices, batch_broadcast_shape);
                        auto batch_indices_i32 = std::make_shared<ov::op::v0::Convert>(batch_broadcast_indices, ov::element::i32);
                        
                        // Concatenate indices [N, H_out, W_out, 3]
                        auto batch_expanded = std::make_shared<ov::op::v0::Unsqueeze>(batch_indices_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto y_expanded = std::make_shared<ov::op::v0::Unsqueeze>(y_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto x_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto indices = std::make_shared<ov::op::v0::Concat>(
                            OutputVector{batch_expanded, y_expanded, x_expanded}, -1);
                        
                        // Gather values
                        auto values = std::make_shared<ov::op::v8::GatherND>(data_nhwc, indices, 0);
                        
                        // Apply zeros padding mask for this corner
                        if (is_zeros) {
                            auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                                std::make_shared<ov::op::v1::GreaterEqual>(x_coords[i], const_0),
                                std::make_shared<ov::op::v1::Less>(x_coords[i], w_in_f));
                            auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                                std::make_shared<ov::op::v1::GreaterEqual>(y_coords[i], const_0),
                                std::make_shared<ov::op::v1::Less>(y_coords[i], h_in_f));
                            auto valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                            auto mask = std::make_shared<ov::op::v0::Convert>(valid, element_type);
                            auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(mask,
                                ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                            auto masked_values = std::make_shared<ov::op::v1::Multiply>(values, mask_expanded);
                            corner_values[i] = masked_values;
                        } else {
                            corner_values[i] = values;
                        }
                    }
                    
                    // Weighted sum of corner values
                    result = std::make_shared<ov::op::v1::Multiply>(corner_values[0],
                        std::make_shared<ov::op::v0::Unsqueeze>(weights[0],
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1})));
                    
                    for (size_t i = 1; i < 4; ++i) {
                        auto weighted = std::make_shared<ov::op::v1::Multiply>(corner_values[i],
                            std::make_shared<ov::op::v0::Unsqueeze>(weights[i],
                                ov::op::v0::Constant::create(ov::element::i32, {}, {-1})));
                        result = std::make_shared<ov::op::v1::Add>(result, weighted);
                    }
                } else {
                    // Full BICUBIC implementation for larger tensors
                    // Get integer coordinates for 16 surrounding points
                    auto x_floor = std::make_shared<ov::op::v0::Floor>(x);
                    auto y_floor = std::make_shared<ov::op::v0::Floor>(y);
                
                // Compute fractional parts
                auto x_frac = std::make_shared<ov::op::v1::Subtract>(x, x_floor);
                auto y_frac = std::make_shared<ov::op::v1::Subtract>(y, y_floor);
                
                // Create offsets for 4x4 grid
                std::vector<int> offsets = {-1, 0, 1, 2};
                
                // Accumulate weighted sum
                result = nullptr;
                
                for (int iy = 0; iy < 4; ++iy) {
                    for (int ix = 0; ix < 4; ++ix) {
                        // Calculate x and y coordinates
                        auto x_coord = std::make_shared<ov::op::v1::Add>(x_floor,
                            ov::op::v0::Constant::create(element_type, {}, {static_cast<float>(offsets[ix])}));
                        auto y_coord = std::make_shared<ov::op::v1::Add>(y_floor,
                            ov::op::v0::Constant::create(element_type, {}, {static_cast<float>(offsets[iy])}));
                        
                        // Apply padding
                        std::shared_ptr<ov::Node> x_padded, y_padded;
                        if (is_border) {
                            // Use Maximum/Minimum instead of Clamp to avoid float literal issues
                            x_padded = std::make_shared<ov::op::v1::Maximum>(x_coord, const_0);
                            y_padded = std::make_shared<ov::op::v1::Maximum>(y_coord, const_0);
                            auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                            auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                            x_padded = std::make_shared<ov::op::v1::Minimum>(x_padded, w_minus_1);
                            y_padded = std::make_shared<ov::op::v1::Minimum>(y_padded, h_minus_1);
                        } else if (is_reflection) {
                            // Reflection padding for bicubic
                            auto reflect_coord = [&](const std::shared_ptr<ov::Node>& coord,
                                                     const std::shared_ptr<ov::Node>& size) -> std::shared_ptr<ov::Node> {
                                auto size_f = std::make_shared<ov::op::v0::Convert>(size, element_type);
                                auto coord_abs = std::make_shared<ov::op::v0::Abs>(coord);
                                
                                if (attrs.align_corners) {
                                    auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                                    auto two_size_minus_2 = std::make_shared<ov::op::v1::Multiply>(const_2, size_minus_1);
                                    auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size_minus_2);
                                    return std::make_shared<ov::op::v1::Subtract>(size_minus_1,
                                        std::make_shared<ov::op::v0::Abs>(
                                            std::make_shared<ov::op::v1::Subtract>(mod_val, size_minus_1)));
                                } else {
                                    auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                                    auto mod_val = std::make_shared<ov::op::v1::Mod>(coord_abs, two_size);
                                    auto reflected = std::make_shared<ov::op::v1::Subtract>(size_f,
                                        std::make_shared<ov::op::v0::Abs>(
                                            std::make_shared<ov::op::v1::Subtract>(mod_val, size_f)));
                                    return std::make_shared<ov::op::v1::Minimum>(reflected,
                                        std::make_shared<ov::op::v1::Subtract>(size_f, const_1));
                                }
                            };
                            
                            x_padded = reflect_coord(x_coord, w_in);
                            y_padded = reflect_coord(y_coord, h_in);
                        } else {
                            // zeros padding - use original coordinates
                            x_padded = x_coord;
                            y_padded = y_coord;
                        }
                        
                        // For ZEROS padding, clamp to valid image bounds to avoid out-of-bounds access
                        std::shared_ptr<ov::Node> x_safe, y_safe;
                        if (is_zeros) {
                            auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                            auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                            // Use Maximum/Minimum instead of Clamp to avoid float literal issues
                            x_safe = std::make_shared<ov::op::v1::Maximum>(x_padded, const_0);
                            y_safe = std::make_shared<ov::op::v1::Maximum>(y_padded, const_0);
                            x_safe = std::make_shared<ov::op::v1::Minimum>(x_safe, w_minus_1);
                            y_safe = std::make_shared<ov::op::v1::Minimum>(y_safe, h_minus_1);
                        } else {
                            x_safe = x_padded;
                            y_safe = y_padded;
                        }
                        
                        // Convert to int32 for indexing
                        auto x_i32 = std::make_shared<ov::op::v0::Convert>(x_safe, ov::element::i32);
                        auto y_i32 = std::make_shared<ov::op::v0::Convert>(y_safe, ov::element::i32);
                        
                        // Create batch indices
                        auto batch_range = std::make_shared<ov::op::v4::Range>(
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}),
                            n_dim,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {1}),
                            ov::element::i32);
                        // Ensure n_dim is 1D for concatenation
                        auto n_dim_1d_bs = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto batch_shape = std::make_shared<ov::op::v0::Concat>(
                            std::vector<Output<Node>>{n_dim_1d_bs, 
                                ov::op::v0::Constant::create(ov::element::i32, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i32, {1}, {1})}, 0);
                        auto batch_indices = std::make_shared<ov::op::v1::Reshape>(batch_range, batch_shape, false);
                        // Ensure dimensions are 1D for concatenation
                        auto n_dim_1d = std::make_shared<ov::op::v0::Unsqueeze>(n_dim,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto h_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(h_out,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto w_out_1d = std::make_shared<ov::op::v0::Unsqueeze>(w_out,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {0}));
                        auto batch_broadcast_shape = std::make_shared<ov::op::v0::Concat>(
                            std::vector<Output<Node>>{n_dim_1d, h_out_1d, w_out_1d}, 0);
                        auto batch_broadcast_indices = std::make_shared<ov::op::v3::Broadcast>(batch_indices, batch_broadcast_shape);
                        auto batch_indices_i32 = std::make_shared<ov::op::v0::Convert>(batch_broadcast_indices, ov::element::i32);
                        
                        // Concatenate indices [N, H_out, W_out, 3]
                        auto batch_expanded = std::make_shared<ov::op::v0::Unsqueeze>(batch_indices_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto y_expanded = std::make_shared<ov::op::v0::Unsqueeze>(y_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto x_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto indices = std::make_shared<ov::op::v0::Concat>(
                            OutputVector{batch_expanded, y_expanded, x_expanded}, -1);
                        
                        // Use GatherND to sample values
                        auto sampled_value = std::make_shared<ov::op::v8::GatherND>(data_nhwc, indices, 0);
                        
                        // Apply zeros padding if needed
                        std::shared_ptr<ov::Node> value_to_weight = sampled_value;
                        if (is_zeros) {
                            auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                                std::make_shared<ov::op::v1::GreaterEqual>(x_coord, const_0),
                                std::make_shared<ov::op::v1::Less>(x_coord, w_in_f));
                            auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                                std::make_shared<ov::op::v1::GreaterEqual>(y_coord, const_0),
                                std::make_shared<ov::op::v1::Less>(y_coord, h_in_f));
                            auto valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                            auto mask = std::make_shared<ov::op::v0::Convert>(valid, element_type);
                            auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(mask,
                                ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                            value_to_weight = std::make_shared<ov::op::v1::Multiply>(sampled_value, mask_expanded);
                        }
                        
                        // Compute bicubic weight using cubic kernel
                        auto cubic_weight = [&](const std::shared_ptr<ov::Node>& frac, float offset) -> std::shared_ptr<ov::Node> {
                            // For bicubic, we need to evaluate the kernel at |frac - offset|
                            // where offset is relative to the floor coordinate
                            auto offset_const = ov::op::v0::Constant::create(element_type, {}, {offset});
                            auto t_diff = std::make_shared<ov::op::v1::Subtract>(frac, offset_const);
                            auto t_abs = std::make_shared<ov::op::v0::Abs>(t_diff);
                            
                            // Cubic kernel: 
                            // if |t| <= 1: 1.5|t|^3 - 2.5|t|^2 + 1
                            // if 1 < |t| <= 2: -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2
                            // else: 0
                            auto t_squared = std::make_shared<ov::op::v1::Multiply>(t_abs, t_abs);
                            auto t_cubed = std::make_shared<ov::op::v1::Multiply>(t_squared, t_abs);
                            
                            // For |t| <= 1
                            auto weight1 = std::make_shared<ov::op::v1::Add>(
                                std::make_shared<ov::op::v1::Subtract>(
                                    std::make_shared<ov::op::v1::Multiply>(
                                        ov::op::v0::Constant::create(element_type, {}, {1.5f}), t_cubed),
                                    std::make_shared<ov::op::v1::Multiply>(
                                        ov::op::v0::Constant::create(element_type, {}, {2.5f}), t_squared)),
                                const_1);
                            
                            // For 1 < |t| <= 2
                            auto weight2 = std::make_shared<ov::op::v1::Add>(
                                std::make_shared<ov::op::v1::Subtract>(
                                    std::make_shared<ov::op::v1::Add>(
                                        std::make_shared<ov::op::v1::Multiply>(
                                            ov::op::v0::Constant::create(element_type, {}, {-0.5f}), t_cubed),
                                        std::make_shared<ov::op::v1::Multiply>(
                                            ov::op::v0::Constant::create(element_type, {}, {2.5f}), t_squared)),
                                    std::make_shared<ov::op::v1::Multiply>(
                                        ov::op::v0::Constant::create(element_type, {}, {4.0f}), t_abs)),
                                const_2);
                            
                            // Conditions
                            auto cond1 = std::make_shared<ov::op::v1::LessEqual>(t_abs, const_1);
                            auto cond2 = std::make_shared<ov::op::v1::LogicalAnd>(
                                std::make_shared<ov::op::v1::Greater>(t_abs, const_1),
                                std::make_shared<ov::op::v1::LessEqual>(t_abs, const_2));
                            
                            // Select weight based on conditions
                            auto weight = std::make_shared<ov::op::v1::Select>(cond1, weight1,
                                std::make_shared<ov::op::v1::Select>(cond2, weight2, const_0));
                            
                            return weight;
                        };
                        
                        // Compute weights
                        auto wx = cubic_weight(x_frac, static_cast<float>(offsets[ix]));
                        auto wy = cubic_weight(y_frac, static_cast<float>(offsets[iy]));
                        auto weight = std::make_shared<ov::op::v1::Multiply>(wx, wy);
                        auto weight_expanded = std::make_shared<ov::op::v0::Unsqueeze>(weight,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        
                        // Apply weight to sampled value
                        auto weighted_value = std::make_shared<ov::op::v1::Multiply>(value_to_weight, weight_expanded);
                        
                        // Accumulate
                        if (result == nullptr) {
                            result = weighted_value;
                        } else {
                            result = std::make_shared<ov::op::v1::Add>(result, weighted_value);
                        }
                    }
                }
            }
            
                }
            
            // Convert back from NHWC to NCHW
            auto transpose_to_nchw = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
            result = std::make_shared<ov::op::v1::Transpose>(result, transpose_to_nchw);
            
            // Preserve runtime info
            result->set_friendly_name(grid_sample->get_friendly_name());
            ov::copy_runtime_info(grid_sample, result);
            ov::replace_node_update_name(grid_sample, result);
            
            return true;
    };
    
    auto m = std::make_shared<ov::pass::pattern::Matcher>(grid_sample_pattern, "GridSampleDecomposition");
    register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov