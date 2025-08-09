// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/grid_sample_decomposition.hpp"

#include <unordered_set>
#include <deque>

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
#include "openvino/op/floor_mod.hpp"
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
#include "snippets/pass/tokenization.hpp"

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
            auto const_0 = ov::op::v0::Constant::create(calc_type, {}, {0.0F});
            auto const_1 = ov::op::v0::Constant::create(calc_type, {}, {1.0F});
            auto const_2 = ov::op::v0::Constant::create(calc_type, {}, {2.0F});
            auto const_0_5 = ov::op::v0::Constant::create(calc_type, {}, {0.5F});
            
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
            
            // Helper function to create batch indices for GatherND operations
            auto create_batch_indices = [&]() {
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
                auto batch_broadcast = std::make_shared<ov::op::v3::Broadcast>(batch_indices, batch_broadcast_shape);
                return std::make_shared<ov::op::v0::Convert>(batch_broadcast, ov::element::i32);
            };
            
            // Helper function to create indices for GatherND and sample values
            auto gather_values = [&](const std::shared_ptr<ov::Node>& x_coord, 
                                    const std::shared_ptr<ov::Node>& y_coord) -> std::shared_ptr<ov::Node> {
                auto x_i32 = std::make_shared<ov::op::v0::Convert>(x_coord, ov::element::i32);
                auto y_i32 = std::make_shared<ov::op::v0::Convert>(y_coord, ov::element::i32);
                auto batch_indices_i32 = create_batch_indices();
                
                auto batch_expanded = std::make_shared<ov::op::v0::Unsqueeze>(batch_indices_i32,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                auto y_expanded = std::make_shared<ov::op::v0::Unsqueeze>(y_i32,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                auto x_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_i32,
                    ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                auto indices = std::make_shared<ov::op::v0::Concat>(
                    OutputVector{batch_expanded, y_expanded, x_expanded}, -1);
                
                return std::make_shared<ov::op::v8::GatherND>(data_nhwc, indices, 0);
            };
            
            // Periodic reflection function for continuous coordinates
            // Handles both align_corners=true and align_corners=false with proper formulas
            auto reflect_coord = [&](const std::shared_ptr<ov::Node>& coord,
                                    const std::shared_ptr<ov::Node>& size_f,
                                    bool align_corners) -> std::shared_ptr<ov::Node> {
                // Special handling for size==1: always return 0
                auto size_is_one = std::make_shared<ov::op::v1::Equal>(size_f, const_1);
                
                if (align_corners) {
                    // align_corners=true: period = 2 * (size - 1)
                    // Correct formula: (size - 1) - abs((coord % period) - (size - 1))
                    auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                    auto period = std::make_shared<ov::op::v1::Multiply>(const_2, size_minus_1);
                    
                    // Use Maximum to avoid division by zero when size==1 (period==0)
                    auto period_safe = std::make_shared<ov::op::v1::Maximum>(period, const_1);
                    
                    auto r = std::make_shared<ov::op::v1::FloorMod>(coord, period_safe);
                    auto d = std::make_shared<ov::op::v1::Subtract>(r, size_minus_1);
                    auto abs_d = std::make_shared<ov::op::v0::Abs>(d);
                    auto reflected = std::make_shared<ov::op::v1::Subtract>(size_minus_1, abs_d);
                    
                    return std::make_shared<ov::op::v1::Select>(size_is_one, const_0, reflected);
                } else {
                    // align_corners=false: reflect around [-0.5, size-0.5], period = 2*size
                    auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                    
                    // Shift so the interval starts at 0: x' ∈ [0, size]
                    auto shifted = std::make_shared<ov::op::v1::Add>(coord, const_0_5);
                    
                    // Modulo into [0, 2*size)
                    auto mod = std::make_shared<ov::op::v1::FloorMod>(shifted, two_size);
                    
                    // Reflect the second half back: if mod > size, use 2*size - mod
                    auto need_reflect = std::make_shared<ov::op::v1::Greater>(mod, size_f);
                    auto reflected_val = std::make_shared<ov::op::v1::Subtract>(two_size, mod);
                    auto reflected = std::make_shared<ov::op::v1::Select>(need_reflect, reflected_val, mod);
                    
                    // Shift back to [-0.5, size-0.5]
                    auto result = std::make_shared<ov::op::v1::Subtract>(reflected, const_0_5);
                    
                    // Size==1 special case stays the same
                    return std::make_shared<ov::op::v1::Select>(size_is_one, const_0, result);
                }
            };
            
            // Helper for discrete index reflection (used after floor/ceil operations)
            auto reflect_index_i32 = [&](const std::shared_ptr<ov::Node>& idx,
                                        const std::shared_ptr<ov::Node>& size_f,
                                        bool align_corners) -> std::shared_ptr<ov::Node> {
                auto size_is_one = std::make_shared<ov::op::v1::Equal>(size_f, const_1);
                
                if (align_corners) {
                    // align_corners=true: reflect using modulo 2*(size-1)
                    auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                    auto two_size_minus_2 = std::make_shared<ov::op::v1::Multiply>(const_2, size_minus_1);
                    
                    // Apply absolute value first
                    auto abs_idx = std::make_shared<ov::op::v0::Abs>(idx);
                    
                    // r = abs_idx % two_size_minus_2
                    auto r = std::make_shared<ov::op::v1::FloorMod>(abs_idx, two_size_minus_2);
                    
                    // reflected = (r >= size) ? (two_size_minus_2 - r) : r
                    auto flipped = std::make_shared<ov::op::v1::Subtract>(two_size_minus_2, r);
                    auto cond = std::make_shared<ov::op::v1::GreaterEqual>(r, size_f);
                    auto reflected = std::make_shared<ov::op::v1::Select>(cond, flipped, r);
                    
                    return std::make_shared<ov::op::v1::Select>(size_is_one, const_0, reflected);
                } else {
                    // align_corners=false: period = 2 * size
                    auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                    auto period_minus_1 = std::make_shared<ov::op::v1::Subtract>(two_size, const_1);
                    
                    // Proper modulo: ((idx mod period) + period) mod period
                    auto mod1 = std::make_shared<ov::op::v1::FloorMod>(idx, two_size);
                    auto mod1_plus_period = std::make_shared<ov::op::v1::Add>(mod1, two_size);
                    auto mod2 = std::make_shared<ov::op::v1::FloorMod>(mod1_plus_period, two_size);
                    
                    // If mod2 >= size, use period_minus_1 - mod2
                    auto need_reflect = std::make_shared<ov::op::v1::GreaterEqual>(mod2, size_f);
                    auto reflected_val = std::make_shared<ov::op::v1::Subtract>(period_minus_1, mod2);
                    auto result = std::make_shared<ov::op::v1::Select>(need_reflect, reflected_val, mod2);
                    
                    return std::make_shared<ov::op::v1::Select>(size_is_one, const_0, result);
                }
            };
            
            std::shared_ptr<ov::Node> result;
            
            if (is_nearest) {
                // NEAREST interpolation
                std::shared_ptr<ov::Node> x_idx, y_idx;
                
                if (is_reflection) {
                    // For REFLECTION mode with NEAREST: round first, then reflect the discrete index
                    // This matches PyTorch behavior where reflection happens on integer indices
                    auto x_rounded = std::make_shared<ov::op::v5::Round>(x, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
                    auto y_rounded = std::make_shared<ov::op::v5::Round>(y, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
                    
                    // Apply reflection to the rounded indices
                    x_idx = reflect_index_i32(x_rounded, w_in_f, attrs.align_corners);
                    y_idx = reflect_index_i32(y_rounded, h_in_f, attrs.align_corners);
                    // No clamping needed - reflection already ensures indices are in [0, size-1]
                } else {
                    // BORDER and ZEROS modes: apply padding/clamping before rounding
                    std::shared_ptr<ov::Node> x_padded = x, y_padded = y;
                    
                    if (is_border) {
                        // Clamp continuous coordinates to [0, size-1] before rounding
                        auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                        auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                        x_padded = std::make_shared<ov::op::v1::Maximum>(x_padded, const_0);
                        x_padded = std::make_shared<ov::op::v1::Minimum>(x_padded, w_minus_1);
                        y_padded = std::make_shared<ov::op::v1::Maximum>(y_padded, const_0);
                        y_padded = std::make_shared<ov::op::v1::Minimum>(y_padded, h_minus_1);
                    }
                    // For ZEROS: use original coordinates, will apply mask later
                    
                    // Round to nearest integer
                    x_idx = std::make_shared<ov::op::v5::Round>(x_padded, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
                    y_idx = std::make_shared<ov::op::v5::Round>(y_padded, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
                    
                    // For BORDER: additional clamp after rounding to handle edge cases
                    if (is_border) {
                        auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                        auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                        x_idx = std::make_shared<ov::op::v1::Maximum>(x_idx, const_0);
                        x_idx = std::make_shared<ov::op::v1::Minimum>(x_idx, w_minus_1);
                        y_idx = std::make_shared<ov::op::v1::Maximum>(y_idx, const_0);
                        y_idx = std::make_shared<ov::op::v1::Minimum>(y_idx, h_minus_1);
                    }
                }
                
                // Use helper function to gather values
                result = gather_values(x_idx, y_idx);
                
                // Apply zeros padding mask if needed
                if (is_zeros) {
                    // Check if rounded indices are within bounds
                    // Convert to int32 first to avoid floating point comparison issues with f16
                    auto x_idx_i32 = std::make_shared<ov::op::v0::Convert>(x_idx, ov::element::i32);
                    auto y_idx_i32 = std::make_shared<ov::op::v0::Convert>(y_idx, ov::element::i32);
                    auto w_i32 = std::make_shared<ov::op::v0::Convert>(w_in_f, ov::element::i32);
                    auto h_i32 = std::make_shared<ov::op::v0::Convert>(h_in_f, ov::element::i32);
                    auto zero_i32 = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
                    
                    auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                        std::make_shared<ov::op::v1::GreaterEqual>(x_idx_i32, zero_i32),
                        std::make_shared<ov::op::v1::Less>(x_idx_i32, w_i32));
                    auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                        std::make_shared<ov::op::v1::GreaterEqual>(y_idx_i32, zero_i32),
                        std::make_shared<ov::op::v1::Less>(y_idx_i32, h_i32));
                    auto valid = std::make_shared<ov::op::v1::LogicalAnd>(x_valid, y_valid);
                    // Convert mask to match result type
                    auto mask = std::make_shared<ov::op::v0::Convert>(valid, element_type);
                    auto mask_expanded = std::make_shared<ov::op::v0::Unsqueeze>(mask,
                        ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                    result = std::make_shared<ov::op::v1::Multiply>(result, mask_expanded);
                }
                
            } else if (is_bilinear) {
                // BILINEAR interpolation
                // Apply padding to continuous coordinates first
                std::shared_ptr<ov::Node> x_padded, y_padded;
                if (is_border) {
                    // For BORDER mode: don't clamp continuous coordinates
                    // The discrete indices will be clamped later
                    // This allows proper interpolation weights at boundaries
                    x_padded = x;
                    y_padded = y;
                } else if (is_reflection) {
                    // For REFLECTION mode: apply periodic reflection to continuous coordinates
                    // This ensures proper behavior for both align_corners variants
                    x_padded = reflect_coord(x, w_in_f, attrs.align_corners);
                    y_padded = reflect_coord(y, h_in_f, attrs.align_corners);
                } else {
                    // zeros padding - use original coordinates
                    x_padded = x;
                    y_padded = y;
                }
                
                // Now compute discrete coordinates and weights from transformed continuous coordinates
                std::shared_ptr<ov::Node> x0 = std::make_shared<ov::op::v0::Floor>(x_padded);
                std::shared_ptr<ov::Node> y0 = std::make_shared<ov::op::v0::Floor>(y_padded);
                std::shared_ptr<ov::Node> x1 = std::make_shared<ov::op::v1::Add>(x0, const_1);
                std::shared_ptr<ov::Node> y1 = std::make_shared<ov::op::v1::Add>(y0, const_1);
                
                // Compute interpolation weights from the reflected continuous coordinates
                // This ensures weights are correctly computed from the reflected positions
                auto dx = std::make_shared<ov::op::v1::Subtract>(x_padded, x0);
                auto dy = std::make_shared<ov::op::v1::Subtract>(y_padded, y0);
                auto one_minus_dx = std::make_shared<ov::op::v1::Subtract>(const_1, dx);
                auto one_minus_dy = std::make_shared<ov::op::v1::Subtract>(const_1, dy);
                
                // Compute weights for 4 corners using dx, dy
                std::vector<std::shared_ptr<ov::Node>> weights;
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(one_minus_dx, one_minus_dy)); // w00
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(dx, one_minus_dy)); // w01
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(one_minus_dx, dy)); // w10
                weights.push_back(std::make_shared<ov::op::v1::Multiply>(dx, dy)); // w11
                
                // For reflection mode, apply discrete index reflection
                if (is_reflection) {
                    // Reflect discrete indices to handle edge cases where floor(x_padded) < 0 or > size-1
                    x0 = reflect_index_i32(x0, w_in_f, attrs.align_corners);
                    x1 = reflect_index_i32(x1, w_in_f, attrs.align_corners);
                    y0 = reflect_index_i32(y0, h_in_f, attrs.align_corners);
                    y1 = reflect_index_i32(y1, h_in_f, attrs.align_corners);
                }
                
                // Store corner indices (now properly reflected if needed)
                std::vector<std::shared_ptr<ov::Node>> x_coords = {x0, x1, x0, x1};
                std::vector<std::shared_ptr<ov::Node>> y_coords = {y0, y0, y1, y1};
                
                // Gather values for each corner
                std::vector<std::shared_ptr<ov::Node>> corner_values(4);
                
                for (size_t i = 0; i < 4; ++i) {
                    auto x_coord = x_coords[i];
                    auto y_coord = y_coords[i];
                    
                    // For zeros and border padding, clamp coordinates for safe indexing
                    if (is_zeros || is_border) {
                        auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                        auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                        x_coord = std::make_shared<ov::op::v1::Maximum>(x_coord, const_0);
                        y_coord = std::make_shared<ov::op::v1::Maximum>(y_coord, const_0);
                        x_coord = std::make_shared<ov::op::v1::Minimum>(x_coord, w_minus_1);
                        y_coord = std::make_shared<ov::op::v1::Minimum>(y_coord, h_minus_1);
                    }
                    // For reflection padding, coordinates are already correctly reflected
                    
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
                    
                    // Apply zeros padding mask for this corner based on original (unpadded) coordinates
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
                // BICUBIC interpolation - proper implementation for ARM
                
                // Step 1: Apply padding/reflection to continuous coordinates FIRST
                std::shared_ptr<ov::Node> x_padded, y_padded;
                
                // Early shortcuts for size==1 axes
                auto w_is_one = std::make_shared<ov::op::v1::Equal>(w_in_f, const_1);
                auto h_is_one = std::make_shared<ov::op::v1::Equal>(h_in_f, const_1);
                
                if (is_border) {
                    // BORDER padding: DO NOT clamp continuous coordinates for BICUBIC
                    // The clamping should only be applied to discrete indices
                    // This allows proper cubic interpolation near boundaries
                    x_padded = x;
                    y_padded = y;
                } else if (is_reflection) {
                    // REFLECTION padding
                    if (!attrs.align_corners) {
                        // align_corners=false: DO NOT reflect continuous coordinates
                        // Just use them as-is, reflection will be applied to discrete indices later
                        x_padded = x;
                        y_padded = y;
                    } else {
                        // align_corners=true: use periodic reflection
                        x_padded = reflect_coord(x, w_in_f, attrs.align_corners);
                        y_padded = reflect_coord(y, h_in_f, attrs.align_corners);
                    }
                } else {
                    // ZEROS padding - NEVER clamp continuous coordinates
                    x_padded = x;
                    y_padded = y;
                }
                
                // Step 2: Compute floor and fractional parts from padded continuous coords
                auto x0 = std::make_shared<ov::op::v0::Floor>(x_padded);
                auto y0 = std::make_shared<ov::op::v0::Floor>(y_padded);
                
                auto dx_raw = std::make_shared<ov::op::v1::Subtract>(x_padded, x0);
                auto dy_raw = std::make_shared<ov::op::v1::Subtract>(y_padded, y0);
                
                // Step 3: Stabilize dx, dy
                std::shared_ptr<ov::Node> dx, dy;
                if (is_zeros) {
                    // For ZEROS padding: don't apply eps clamping, use raw fractional parts
                    // This ensures correct weights even at boundaries
                    dx = dx_raw;
                    dy = dy_raw;
                } else {
                    // For BORDER/REFLECTION: apply eps clamping for numerical stability
                    auto eps = ov::op::v0::Constant::create(calc_type, {}, {1e-7F});
                    auto one_minus_eps = std::make_shared<ov::op::v1::Subtract>(const_1, eps);
                    dx = std::make_shared<ov::op::v1::Maximum>(dx_raw, const_0);
                    dx = std::make_shared<ov::op::v1::Minimum>(dx, one_minus_eps);
                    dy = std::make_shared<ov::op::v1::Maximum>(dy_raw, const_0);
                    dy = std::make_shared<ov::op::v1::Minimum>(dy, one_minus_eps);
                }
                
                // For size==1: force dx=0 or dy=0
                dx = std::make_shared<ov::op::v1::Select>(w_is_one, const_0, dx);
                dy = std::make_shared<ov::op::v1::Select>(h_is_one, const_0, dy);
                
                // Implement proper BICUBIC with unified cubic polynomials
                // Cubic coefficients a = -0.75
                auto a = ov::op::v0::Constant::create(calc_type, {}, {-0.75F});
                auto const_3 = ov::op::v0::Constant::create(calc_type, {}, {3.0F});
                auto const_4 = ov::op::v0::Constant::create(calc_type, {}, {4.0F}); 
                auto const_5 = ov::op::v0::Constant::create(calc_type, {}, {5.0F});
                auto const_8 = ov::op::v0::Constant::create(calc_type, {}, {8.0F});
                
                // Helper function for cubic polynomial evaluation
                auto cubic_weight = [&](const std::shared_ptr<ov::Node>& t) -> std::shared_ptr<ov::Node> {
                    // Clamp t to [0, 2] to ensure numerical stability
                    std::shared_ptr<ov::Node> t_clamped = std::make_shared<ov::op::v1::Maximum>(t, const_0);
                    t_clamped = std::make_shared<ov::op::v1::Minimum>(t_clamped, const_2);
                    
                    auto t_sq = std::make_shared<ov::op::v1::Multiply>(t_clamped, t_clamped);
                    auto t_cb = std::make_shared<ov::op::v1::Multiply>(t_sq, t_clamped);
                    
                    // Branch 1: 0 <= t <= 1: f01(t) = ((a+2)*t - (a+3))*t*t + 1
                    auto a_plus_2 = std::make_shared<ov::op::v1::Add>(a, const_2);
                    auto a_plus_3 = std::make_shared<ov::op::v1::Add>(a, const_3);
                    auto f01_inner = std::make_shared<ov::op::v1::Subtract>(
                        std::make_shared<ov::op::v1::Multiply>(a_plus_2, t_clamped), a_plus_3);
                    auto f01 = std::make_shared<ov::op::v1::Add>(
                        std::make_shared<ov::op::v1::Multiply>(f01_inner, t_sq), const_1);
                    
                    // Branch 2: 1 < t < 2: f12(t) = ((a*t - 5a)*t + 8a)*t - 4a
                    auto five_a = std::make_shared<ov::op::v1::Multiply>(const_5, a);
                    auto eight_a = std::make_shared<ov::op::v1::Multiply>(const_8, a);
                    auto four_a = std::make_shared<ov::op::v1::Multiply>(const_4, a);
                    auto f12_inner1 = std::make_shared<ov::op::v1::Subtract>(
                        std::make_shared<ov::op::v1::Multiply>(a, t_clamped), five_a);
                    auto f12_inner2 = std::make_shared<ov::op::v1::Add>(
                        std::make_shared<ov::op::v1::Multiply>(f12_inner1, t_clamped), eight_a);
                    auto f12 = std::make_shared<ov::op::v1::Subtract>(
                        std::make_shared<ov::op::v1::Multiply>(f12_inner2, t_clamped), four_a);
                    
                    // Conditions - use strict bounds: t <= 1 and t < 2
                    auto cond1 = std::make_shared<ov::op::v1::LessEqual>(t_clamped, const_1);
                    auto cond2 = std::make_shared<ov::op::v1::Less>(t_clamped, const_2);  // Strict: t < 2, not t <= 2
                    
                    // Select appropriate branch
                    auto weight = std::make_shared<ov::op::v1::Select>(cond1, f01,
                        std::make_shared<ov::op::v1::Select>(cond2, f12, const_0));
                    
                    return weight;
                };
                
                // Calculate 4 cubic weights for X direction
                auto t_x0 = std::make_shared<ov::op::v1::Add>(const_1, dx);  // 1 + dx
                auto t_x1 = dx;                                              // dx  
                auto t_x2 = std::make_shared<ov::op::v1::Subtract>(const_1, dx);  // 1 - dx
                auto t_x3 = std::make_shared<ov::op::v1::Subtract>(const_2, dx);  // 2 - dx
                
                auto w_x0 = cubic_weight(t_x0);
                auto w_x1 = cubic_weight(t_x1); 
                auto w_x2 = cubic_weight(t_x2);
                auto w_x3 = cubic_weight(t_x3);
                
                // Force exact weights for W_in==1: [0,1,0,0]
                w_x0 = std::make_shared<ov::op::v1::Select>(w_is_one, const_0, w_x0);
                w_x1 = std::make_shared<ov::op::v1::Select>(w_is_one, const_1, w_x1);
                w_x2 = std::make_shared<ov::op::v1::Select>(w_is_one, const_0, w_x2);
                w_x3 = std::make_shared<ov::op::v1::Select>(w_is_one, const_0, w_x3);
                
                // Calculate 4 cubic weights for Y direction  
                auto t_y0 = std::make_shared<ov::op::v1::Add>(const_1, dy);  // 1 + dy
                auto t_y1 = dy;                                              // dy
                auto t_y2 = std::make_shared<ov::op::v1::Subtract>(const_1, dy);  // 1 - dy
                auto t_y3 = std::make_shared<ov::op::v1::Subtract>(const_2, dy);  // 2 - dy
                
                auto w_y0 = cubic_weight(t_y0);
                auto w_y1 = cubic_weight(t_y1);
                auto w_y2 = cubic_weight(t_y2);
                auto w_y3 = cubic_weight(t_y3);
                
                // Force exact weights for H_in==1: [0,1,0,0]
                w_y0 = std::make_shared<ov::op::v1::Select>(h_is_one, const_0, w_y0);
                w_y1 = std::make_shared<ov::op::v1::Select>(h_is_one, const_1, w_y1);
                w_y2 = std::make_shared<ov::op::v1::Select>(h_is_one, const_0, w_y2);
                w_y3 = std::make_shared<ov::op::v1::Select>(h_is_one, const_0, w_y3);
                
                // Calculate 4x4 discrete indices (offsets from floor)
                std::vector<std::shared_ptr<ov::Node>> x_indices, y_indices;
                std::vector<std::shared_ptr<ov::Node>> x_weights, y_weights;
                
                // X indices and weights
                x_indices.push_back(std::make_shared<ov::op::v1::Subtract>(x0, const_1)); // x0-1
                x_indices.push_back(x0);                                                   // x0  
                x_indices.push_back(std::make_shared<ov::op::v1::Add>(x0, const_1));     // x0+1
                x_indices.push_back(std::make_shared<ov::op::v1::Add>(x0, const_2));     // x0+2
                x_weights = {w_x0, w_x1, w_x2, w_x3};
                
                // Y indices and weights
                y_indices.push_back(std::make_shared<ov::op::v1::Subtract>(y0, const_1)); // y0-1
                y_indices.push_back(y0);                                                   // y0
                y_indices.push_back(std::make_shared<ov::op::v1::Add>(y0, const_1));     // y0+1
                y_indices.push_back(std::make_shared<ov::op::v1::Add>(y0, const_2));     // y0+2
                y_weights = {w_y0, w_y1, w_y2, w_y3};
                
                // Build ZEROS validity mask from raw discrete indices BEFORE clamping
                std::vector<std::shared_ptr<ov::Node>> x_valid_masks, y_valid_masks;
                
                if (is_zeros) {
                    for (int i = 0; i < 4; ++i) {
                        // X validity: 0 <= x_index < W_in
                        auto x_geq_0 = std::make_shared<ov::op::v1::GreaterEqual>(x_indices[i], const_0);
                        auto x_lt_w = std::make_shared<ov::op::v1::Less>(x_indices[i], w_in_f);
                        auto x_valid = std::make_shared<ov::op::v1::LogicalAnd>(x_geq_0, x_lt_w);
                        x_valid_masks.push_back(x_valid);
                        
                        // Y validity: 0 <= y_index < H_in  
                        auto y_geq_0 = std::make_shared<ov::op::v1::GreaterEqual>(y_indices[i], const_0);
                        auto y_lt_h = std::make_shared<ov::op::v1::Less>(y_indices[i], h_in_f);
                        auto y_valid = std::make_shared<ov::op::v1::LogicalAnd>(y_geq_0, y_lt_h);
                        y_valid_masks.push_back(y_valid);
                    }
                }
                
                // Apply discrete index reflection for REFLECTION mode
                // IMPORTANT: For align_corners=false, continuous coordinates are already reflected!
                // We should NOT apply the standard reflection formula again.
                // Instead, just handle boundary cases with clamping and mirroring.
                if (is_reflection) {
                    // Helper lambda for discrete index reflection
                    auto reflect_discrete_index = [&](const std::shared_ptr<ov::Node>& idx, 
                                                      const std::shared_ptr<ov::Node>& size_f,
                                                      const std::shared_ptr<ov::Node>& is_size_one) {
                        // For size==1: always return 0
                        // Otherwise: apply reflection with modulo arithmetic
                        
                        if (!attrs.align_corners) {
                            // align_corners=false: Apply reflection to discrete indices
                            // period = 2 * size
                            // i = ((i mod period) + period) mod period  (proper modulo for negative numbers)
                            // if i >= size, replace with (period - 1 - i)
                            
                            auto two_size = std::make_shared<ov::op::v1::Multiply>(const_2, size_f);
                            auto period_minus_1 = std::make_shared<ov::op::v1::Subtract>(two_size, const_1);
                            
                            // Proper modulo: ((idx mod period) + period) mod period
                            auto mod1 = std::make_shared<ov::op::v1::FloorMod>(idx, two_size);
                            auto mod1_plus_period = std::make_shared<ov::op::v1::Add>(mod1, two_size);
                            auto mod2 = std::make_shared<ov::op::v1::FloorMod>(mod1_plus_period, two_size);
                            
                            // If mod2 >= size, use period_minus_1 - mod2
                            auto need_reflect = std::make_shared<ov::op::v1::GreaterEqual>(mod2, size_f);
                            auto reflected_val = std::make_shared<ov::op::v1::Subtract>(period_minus_1, mod2);
                            auto result = std::make_shared<ov::op::v1::Select>(need_reflect, reflected_val, mod2);
                            
                            return std::make_shared<ov::op::v1::Select>(is_size_one, const_0, result);
                        } else {
                            // align_corners=true: reflect using modulo 2*(size-1)
                            auto size_minus_1 = std::make_shared<ov::op::v1::Subtract>(size_f, const_1);
                            auto two_size_minus_2 = std::make_shared<ov::op::v1::Multiply>(
                                const_2, size_minus_1);
                            
                            // Apply absolute value first
                            auto abs_idx = std::make_shared<ov::op::v0::Abs>(idx);
                            
                            // r = abs_idx % two_size_minus_2
                            auto r = std::make_shared<ov::op::v1::FloorMod>(abs_idx, two_size_minus_2);
                            
                            // reflected = (r >= size) ? (two_size_minus_2 - r) : r
                            auto flipped = std::make_shared<ov::op::v1::Subtract>(two_size_minus_2, r);
                            auto cond = std::make_shared<ov::op::v1::GreaterEqual>(r, size_f);
                            auto reflected = std::make_shared<ov::op::v1::Select>(cond, flipped, r);
                            
                            return std::make_shared<ov::op::v1::Select>(is_size_one, const_0, reflected);
                        }
                    };
                    
                    // Apply reflection to all x and y indices
                    for (int i = 0; i < 4; ++i) {
                        x_indices[i] = reflect_discrete_index(x_indices[i], w_in_f, w_is_one);
                        y_indices[i] = reflect_discrete_index(y_indices[i], h_in_f, h_is_one);
                    }
                } else {
                    // For size==1 cases in non-REFLECTION modes, force indices to 0
                    for (int i = 0; i < 4; ++i) {
                        x_indices[i] = std::make_shared<ov::op::v1::Select>(w_is_one, const_0, x_indices[i]);
                        y_indices[i] = std::make_shared<ov::op::v1::Select>(h_is_one, const_0, y_indices[i]);
                    }
                }
                
                // Sample all 16 values using 4x4 grid
                // Optimized version: process in row-wise fashion to reduce register pressure
                result = nullptr;
                
                // Process each row separately and accumulate
                for (int iy = 0; iy < 4; ++iy) {
                    std::shared_ptr<ov::Node> row_result = nullptr;
                    
                    for (int ix = 0; ix < 4; ++ix) {
                        auto x_idx = x_indices[ix];
                        auto y_idx = y_indices[iy];
                        
                        // For ZEROS: use safe indices for gathering (clamp to valid range)
                        // The actual masking is done via x_valid_masks and y_valid_masks
                        // For BORDER: clamp indices to valid range for border replication
                        // For REFLECTION: indices are already reflected, no clamping needed
                        std::shared_ptr<ov::Node> x_idx_safe = x_idx;
                        std::shared_ptr<ov::Node> y_idx_safe = y_idx;
                        
                        if (is_zeros || is_border) {
                            auto w_minus_1 = std::make_shared<ov::op::v1::Subtract>(w_in_f, const_1);
                            auto h_minus_1 = std::make_shared<ov::op::v1::Subtract>(h_in_f, const_1);
                            x_idx_safe = std::make_shared<ov::op::v1::Maximum>(x_idx, const_0);
                            y_idx_safe = std::make_shared<ov::op::v1::Maximum>(y_idx, const_0);
                            x_idx_safe = std::make_shared<ov::op::v1::Minimum>(x_idx_safe, w_minus_1);
                            y_idx_safe = std::make_shared<ov::op::v1::Minimum>(y_idx_safe, h_minus_1);
                        }
                        
                        // Convert to int32 and gather
                        auto x_i32 = std::make_shared<ov::op::v0::Convert>(x_idx_safe, ov::element::i32);
                        auto y_i32 = std::make_shared<ov::op::v0::Convert>(y_idx_safe, ov::element::i32);
                        
                        // Create batch indices for GatherND
                        auto batch_indices = create_batch_indices();
                        auto batch_expanded = std::make_shared<ov::op::v0::Unsqueeze>(batch_indices,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto y_expanded = std::make_shared<ov::op::v0::Unsqueeze>(y_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto x_expanded = std::make_shared<ov::op::v0::Unsqueeze>(x_i32,
                            ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                        auto indices = std::make_shared<ov::op::v0::Concat>(
                            OutputVector{batch_expanded, y_expanded, x_expanded}, -1);
                        
                        auto sampled_value = std::make_shared<ov::op::v8::GatherND>(data_nhwc, indices, 0);
                        
                        // Calculate combined weight for this tap
                        auto combined_weight = std::make_shared<ov::op::v1::Multiply>(x_weights[ix], y_weights[iy]);
                        
                        // Apply ZEROS mask (multiply by validity of BOTH x and y for this tap)
                        std::shared_ptr<ov::Node> final_value;
                        if (is_zeros) {
                            auto tap_valid = std::make_shared<ov::op::v1::LogicalAnd>(
                                x_valid_masks[ix], y_valid_masks[iy]);
                            auto validity_mask = std::make_shared<ov::op::v0::Convert>(tap_valid, element_type);
                            auto masked_weight = std::make_shared<ov::op::v1::Multiply>(combined_weight, validity_mask);
                            auto weight_expanded = std::make_shared<ov::op::v0::Unsqueeze>(masked_weight,
                                ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                            final_value = std::make_shared<ov::op::v1::Multiply>(sampled_value, weight_expanded);
                        } else {
                            auto weight_expanded = std::make_shared<ov::op::v0::Unsqueeze>(combined_weight,
                                ov::op::v0::Constant::create(ov::element::i32, {}, {-1}));
                            final_value = std::make_shared<ov::op::v1::Multiply>(sampled_value, weight_expanded);
                        }
                        
                        // Accumulate within row first (reduces live tensors)
                        if (row_result == nullptr) {
                            row_result = final_value;
                        } else {
                            row_result = std::make_shared<ov::op::v1::Add>(row_result, final_value);
                        }
                    }
                    
                    // Accumulate row result to final result
                    if (result == nullptr) {
                        result = row_result;
                    } else {
                        result = std::make_shared<ov::op::v1::Add>(result, row_result);
                    }
                }
            } else {
                // Unsupported interpolation mode
                return false;
            }
            
            // Convert back from NHWC to NCHW
            auto transpose_to_nchw = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 3, 1, 2});
            result = std::make_shared<ov::op::v1::Transpose>(result, transpose_to_nchw);
            
            // With optimized row-wise accumulation, we can try to keep Snippets enabled
            // The reduced register pressure should allow Snippets to work
            // If register allocation still fails, uncomment the code below:
            /*
            if (is_bicubic) {
                // Mark all operations in the subgraph to skip Snippets
                std::unordered_set<std::shared_ptr<ov::Node>> visited;
                std::deque<std::shared_ptr<ov::Node>> queue;
                queue.push_back(result);
                
                while (!queue.empty()) {
                    auto node = queue.front();
                    queue.pop_front();
                    
                    if (visited.count(node) > 0)
                        continue;
                    visited.insert(node);
                    
                    // Mark this node to skip Snippets tokenization using the proper API
                    snippets::pass::SetSnippetsNodeType(node, snippets::pass::SnippetsNodeType::SkippedByPlugin);
                    
                    // Process inputs
                    for (auto& input : node->inputs()) {
                        auto input_node = input.get_source_output().get_node_shared_ptr();
                        if (input_node && visited.count(input_node) == 0) {
                            queue.push_back(input_node);
                        }
                    }
                }
            }
            */
            
            // Preserve runtime info
            // Note: result has already been transposed back to NCHW on line 857
            result->set_friendly_name(grid_sample->get_friendly_name());
            ov::copy_runtime_info(grid_sample, result);
            ov::replace_node_update_name(grid_sample, result);
            
            return true;
    };
    
    auto m = std::make_shared<ov::pass::pattern::Matcher>(grid_sample_pattern, "GridSampleDecomposition");
    register_matcher(m, callback);
}

// Register transformation passes for each mode
// These are just empty implementations since all the logic is in GridSampleDecomposition
// We rely on the GridSampleDecomposition to handle all modes
GridSampleDecompositionNearest::GridSampleDecompositionNearest() {
}

GridSampleDecompositionBilinear::GridSampleDecompositionBilinear() {
}

GridSampleDecompositionBicubic::GridSampleDecompositionBicubic() {
}

}  // namespace intel_cpu
}  // namespace ov
