// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv_3d_decomposition.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_cpu {

Deconv3DDecomposition::Deconv3DDecomposition() {
    // Match 3D ConvolutionBackpropData operations
    auto deconv_3d = ov::pass::pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (!ov::is_type<ov::op::v1::ConvolutionBackpropData>(node)) {
            return false;
        }
        auto deconv = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(node);
        if (!deconv) {
            return false;
        }

        // Check if this is a 3D deconvolution
        auto input_shape = deconv->get_input_partial_shape(0);
        auto weights_shape = deconv->get_input_partial_shape(1);

        // Check if this is a 3D deconvolution (rank must be 3)
        if (input_shape.rank().is_dynamic() || weights_shape.rank().is_dynamic()) {
            return false;
        }

        if (input_shape.rank() != 3 || weights_shape.rank() != 3) {
            return false;
        }

        // Get original inputs
        auto input = deconv->input_value(0);
        auto weights = deconv->input_value(1);

        // Get deconv attributes
        auto strides = deconv->get_strides();
        auto pads_begin = deconv->get_pads_begin();
        auto pads_end = deconv->get_pads_end();
        auto dilations = deconv->get_dilations();
        auto auto_pad = deconv->get_auto_pad();

        // Handle auto_pad by converting to explicit padding
        if (auto_pad != ov::op::PadType::EXPLICIT && auto_pad != ov::op::PadType::NOTSET) {
            // For auto_pad modes, we need static shapes to calculate padding
            if (input_shape.is_dynamic() || weights_shape.is_dynamic()) {
                return false;
            }

            // Calculate padding based on auto_pad mode
            // For ConvolutionBackpropData with auto_pad:
            // output_shape = (input_shape - 1) * stride + kernel_size - pad_begin - pad_end
            // For SAME_UPPER/SAME_LOWER: output_shape = input_shape * stride
            // For VALID: no padding

            auto stride = strides[0];

            if (auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::SAME_LOWER) {
                // For SAME padding in deconv: total_padding = stride - 1
                // This ensures output_size = input_size * stride
                auto total_padding = std::max<int64_t>(0, stride - 1);

                if (auto_pad == ov::op::PadType::SAME_UPPER) {
                    pads_begin[0] = total_padding / 2;
                    pads_end[0] = total_padding - pads_begin[0];
                } else {  // SAME_LOWER
                    pads_end[0] = total_padding / 2;
                    pads_begin[0] = total_padding - pads_end[0];
                }
            } else if (auto_pad == ov::op::PadType::VALID) {
                pads_begin[0] = 0;
                pads_end[0] = 0;
            }

            // Continue with explicit padding
            auto_pad = ov::op::PadType::EXPLICIT;
        }
        auto output_padding = deconv->get_output_padding();

        // Only handle 1D deconv (single stride, pad, dilation)
        if (strides.size() != 1 || pads_begin.size() != 1 || pads_end.size() != 1 || dilations.size() != 1) {
            return false;
        }

        // Get parameters
        auto stride = strides[0];
        auto pad_begin = pads_begin[0];
        auto pad_end = pads_end[0];
        auto dilation = dilations[0];
        auto out_pad = output_padding.empty() ? 0 : output_padding[0];

        // Get input dimensions [N, C_in, L]
        auto N = input_shape[0];
        auto C_in = input_shape[1];
        auto L = input_shape[2];

        // Get weights dimensions
        // ConvolutionBackpropData weights have format [C_in, C_out, K] in OpenVINO
        auto C_in_weights = weights_shape[0];
        auto C_out = weights_shape[1];
        auto K = weights_shape[2];

        // For dynamic shapes, we'll need to use ShapeOf operations
        bool is_dynamic = input_shape.is_dynamic() || weights_shape.is_dynamic();

        // Verify channel dimensions match
        if (C_in != C_in_weights) {
            return false;
        }

        ov::Output<ov::Node> result = input;
        std::vector<std::shared_ptr<ov::Node>> decomp_nodes;

        // Helper to create shape patterns for dynamic shapes
        auto create_shape_pattern = [&](const std::vector<ov::Output<ov::Node>>& dims) -> ov::Output<ov::Node> {
            if (dims.size() == 1) {
                return dims[0];
            }
            return std::make_shared<ov::op::v0::Concat>(dims, 0);
        };

        // Get shape components
        ov::Output<ov::Node> shape_of_input;
        ov::Output<ov::Node> N_node, C_in_node, L_node;

        // Create constants we'll need throughout the transformation
        auto one_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto zero_const = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        auto one_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
        auto two_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {2});
        auto minus_one = ov::op::v0::Constant::create(ov::element::i64, {}, {-1});
        // Constants for Slice operation
        auto zero_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto three_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        if (is_dynamic) {
            shape_of_input = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
            auto gather_n = std::make_shared<ov::op::v8::Gather>(shape_of_input, zero_const, zero_const);
            auto gather_c = std::make_shared<ov::op::v8::Gather>(shape_of_input, one_scalar, zero_const);
            auto gather_l = std::make_shared<ov::op::v8::Gather>(shape_of_input, two_scalar, zero_const);

            // Unsqueeze to make them 1D tensors for concatenation
            N_node = std::make_shared<ov::op::v0::Unsqueeze>(gather_n, zero_const);
            C_in_node = std::make_shared<ov::op::v0::Unsqueeze>(gather_c, zero_const);
            L_node = std::make_shared<ov::op::v0::Unsqueeze>(gather_l, zero_const);
        } else {
            // For static shapes, create 1D constants (not scalars) for concatenation
            N_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {N.get_length()});
            C_in_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {C_in.get_length()});
            L_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {L.get_length()});
        }

        // Reshape input from 3D to 4D: [N, C_in, L] -> [N, C_in, L, 1]
        ov::Output<ov::Node> input_4d_pattern;
        if (is_dynamic) {
            input_4d_pattern = create_shape_pattern({N_node, C_in_node, L_node, one_const});
        } else {
            // For static shapes, create a constant directly
            std::vector<int64_t> input_4d_shape = {static_cast<int64_t>(N.get_length()),
                                                   static_cast<int64_t>(C_in.get_length()),
                                                   static_cast<int64_t>(L.get_length()),
                                                   1};
            input_4d_pattern = ov::op::v0::Constant::create(ov::element::i64, {4}, input_4d_shape);
        }
        auto input_4d = std::make_shared<ov::op::v1::Reshape>(input, input_4d_pattern, false);
        result = input_4d;
        decomp_nodes.push_back(input_4d);

        // Step 1: Insert zeros (upsampling) if stride > 1
        if (stride > 1) {
            // New size = L + (L - 1) * (stride - 1) = (L - 1) * stride + 1
            ov::Output<ov::Node> upsampled_L_node;

            if (is_dynamic) {
                // Calculate upsampled_L = (L - 1) * stride + 1
                auto stride_const = ov::op::v0::Constant::create(ov::element::i64, {}, {stride});
                auto L_minus_1 = std::make_shared<ov::op::v1::Add>(L_node, minus_one);
                auto L_minus_1_times_stride = std::make_shared<ov::op::v1::Multiply>(L_minus_1, stride_const);
                upsampled_L_node = std::make_shared<ov::op::v1::Add>(L_minus_1_times_stride, one_scalar);
            } else {
                auto upsampled_L = (L.get_length() - 1) * stride + 1;
                upsampled_L_node = ov::op::v0::Constant::create(ov::element::i64, {}, {upsampled_L});
            }

            // Create a new 4D tensor filled with zeros
            ov::Output<ov::Node> zero_tensor;
            if (is_dynamic) {
                // Create shape [N, C_in, upsampled_L, 1] for zeros
                // Need to unsqueeze scalar constants to 1D tensors for concatenation
                auto one_const_1d = std::make_shared<ov::op::v0::Unsqueeze>(one_const, zero_const);
                auto zero_shape = create_shape_pattern({N_node, C_in_node, upsampled_L_node, one_const_1d});
                // Create zeros with dynamic shape
                auto zero_scalar = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
                zero_tensor = std::make_shared<ov::op::v3::Broadcast>(zero_scalar, zero_shape);
            } else {
                auto upsampled_L = (L.get_length() - 1) * stride + 1;
                ov::Shape upsampled_shape = {static_cast<size_t>(N.get_length()),
                                             static_cast<size_t>(C_in.get_length()),
                                             static_cast<size_t>(upsampled_L),
                                             1};
                zero_tensor = ov::op::v0::Constant::create(result.get_element_type(), upsampled_shape, {0});
            }

            // Create indices for scatter: [0, stride, 2*stride, ...]
            ov::Output<ov::Node> indices;
            if (is_dynamic) {
                // Create range [0, L) and multiply by stride
                auto stride_const = ov::op::v0::Constant::create(ov::element::i64, {}, {stride});
                auto range = std::make_shared<ov::op::v4::Range>(zero_const, L_node, one_scalar, ov::element::i64);
                indices = std::make_shared<ov::op::v1::Multiply>(range, stride_const);
            } else {
                std::vector<int64_t> indices_data;
                for (int64_t i = 0; i < L.get_length(); i++) {
                    indices_data.push_back(static_cast<int64_t>(i * stride));
                }
                indices =
                    ov::op::v0::Constant::create(ov::element::i64, {static_cast<size_t>(L.get_length())}, indices_data);
            }

            // Reshape input to [N*C_in, L, 1] for scatter
            // Mark reshape operations for in-place execution to reduce memory overhead
            ov::Output<ov::Node> input_3d;
            if (is_dynamic) {
                auto N_times_C_in = std::make_shared<ov::op::v1::Multiply>(N_node, C_in_node);
                auto one_const_1d = std::make_shared<ov::op::v0::Unsqueeze>(one_const, zero_const);
                auto input_3d_shape = create_shape_pattern({N_times_C_in, L_node, one_const_1d});
                input_3d = std::make_shared<ov::op::v1::Reshape>(result,
                                                                 input_3d_shape,
                                                                 true);  // special_zero = true for optimization
            } else {
                std::vector<int64_t> reshape_3d = {static_cast<int64_t>(N.get_length() * C_in.get_length()),
                                                   static_cast<int64_t>(L.get_length()),
                                                   1};
                auto input_reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, {3}, reshape_3d);
                input_3d =
                    std::make_shared<ov::op::v1::Reshape>(result, input_reshape_pattern, true);  // special_zero = true
            }

            // Reshape zero tensor to [N*C_in, upsampled_L, 1]
            ov::Output<ov::Node> zero_3d;
            if (is_dynamic) {
                auto N_times_C_in = std::make_shared<ov::op::v1::Multiply>(N_node, C_in_node);
                auto one_const_1d = std::make_shared<ov::op::v0::Unsqueeze>(one_const, zero_const);
                auto zero_3d_shape = create_shape_pattern({N_times_C_in, upsampled_L_node, one_const_1d});
                zero_3d = std::make_shared<ov::op::v1::Reshape>(zero_tensor, zero_3d_shape, true);
            } else {
                auto upsampled_L = (L.get_length() - 1) * stride + 1;
                std::vector<int64_t> zero_reshape_3d = {static_cast<int64_t>(N.get_length() * C_in.get_length()),
                                                        static_cast<int64_t>(upsampled_L),
                                                        1};
                auto zero_reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, {3}, zero_reshape_3d);
                zero_3d = std::make_shared<ov::op::v1::Reshape>(zero_tensor, zero_reshape_pattern, true);
            }

            // Scatter update along axis 1
            auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(zero_3d, indices, input_3d, one_scalar);

            // Reshape back to 4D
            ov::Output<ov::Node> upsampled;
            if (is_dynamic) {
                auto one_const_1d = std::make_shared<ov::op::v0::Unsqueeze>(one_const, zero_const);
                auto upsampled_4d_shape = create_shape_pattern({N_node, C_in_node, upsampled_L_node, one_const_1d});
                upsampled = std::make_shared<ov::op::v1::Reshape>(scatter, upsampled_4d_shape, false);
            } else {
                auto upsampled_L = (L.get_length() - 1) * stride + 1;
                std::vector<int64_t> upsampled_shape_i64 = {static_cast<int64_t>(N.get_length()),
                                                            static_cast<int64_t>(C_in.get_length()),
                                                            static_cast<int64_t>(upsampled_L),
                                                            1};
                auto output_reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, {4}, upsampled_shape_i64);
                upsampled = std::make_shared<ov::op::v1::Reshape>(scatter, output_reshape_pattern, false);
            }

            result = upsampled;
            decomp_nodes.push_back(input_3d.get_node_shared_ptr());
            decomp_nodes.push_back(zero_3d.get_node_shared_ptr());
            decomp_nodes.push_back(scatter);
            decomp_nodes.push_back(upsampled.get_node_shared_ptr());
        }

        // Step 2: Padding (for 4D tensors)
        if (K.is_static()) {
            // Static padding calculation
            int64_t pad_left_val = K.get_length() - 1 - pad_begin;
            int64_t pad_right_val = K.get_length() - 1 - pad_end + out_pad;

            // Ensure non-negative padding
            pad_left_val = std::max<int64_t>(0, pad_left_val);
            pad_right_val = std::max<int64_t>(0, pad_right_val);

            // Skip padding if both values are zero to reduce operations
            if (pad_left_val > 0 || pad_right_val > 0) {
                std::vector<int64_t> pads_begin_vec = {0, 0, pad_left_val, 0};
                std::vector<int64_t> pads_end_vec = {0, 0, pad_right_val, 0};

                auto pads_begin_const = ov::op::v0::Constant::create(ov::element::i64, {4}, pads_begin_vec);
                auto pads_end_const = ov::op::v0::Constant::create(ov::element::i64, {4}, pads_end_vec);
                auto pad_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});

                auto pad_op = std::make_shared<ov::op::v1::Pad>(result,
                                                                pads_begin_const,
                                                                pads_end_const,
                                                                pad_value,
                                                                ov::op::PadMode::CONSTANT);
                result = pad_op;
                decomp_nodes.push_back(pad_op);
            }
        } else {
            // Dynamic padding calculation
            // For dynamic kernel size, we always need to calculate padding
            // because pad_left = K - 1 - pad_begin and pad_right = K - 1 - pad_end + out_pad
            // Even with pad_begin=0, pad_end=0, out_pad=0, we still have pad_left = K-1, pad_right = K-1
            // Only skip if we can prove K=1 (which we can't with dynamic shape)
            auto shape_of_weights = std::make_shared<ov::op::v3::ShapeOf>(weights, ov::element::i64);
            auto K_dyn = std::make_shared<ov::op::v8::Gather>(shape_of_weights, two_scalar, zero_const);

            // Calculate pad_left = K - 1 - pad_begin
            auto pad_begin_const = ov::op::v0::Constant::create(ov::element::i64, {}, {pad_begin});
            auto K_minus_1 = std::make_shared<ov::op::v1::Add>(K_dyn, minus_one);
            auto pad_left_dyn = std::make_shared<ov::op::v1::Subtract>(K_minus_1, pad_begin_const);

            // Calculate pad_right = K - 1 - pad_end + out_pad
            auto pad_end_const = ov::op::v0::Constant::create(ov::element::i64, {}, {pad_end});
            auto out_pad_const = ov::op::v0::Constant::create(ov::element::i64, {}, {out_pad});
            auto K_minus_1_minus_pad_end = std::make_shared<ov::op::v1::Subtract>(K_minus_1, pad_end_const);
            auto pad_right_dyn = std::make_shared<ov::op::v1::Add>(K_minus_1_minus_pad_end, out_pad_const);

            // Create padding vectors [0, 0, pad_left, 0] and [0, 0, pad_right, 0]
            auto zero_1d = std::make_shared<ov::op::v0::Unsqueeze>(zero_const, zero_const);
            auto pad_left_1d = std::make_shared<ov::op::v0::Unsqueeze>(pad_left_dyn, zero_const);
            auto pad_right_1d = std::make_shared<ov::op::v0::Unsqueeze>(pad_right_dyn, zero_const);

            auto pads_begin_dyn =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{zero_1d, zero_1d, pad_left_1d, zero_1d}, 0);
            auto pads_end_dyn =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{zero_1d, zero_1d, pad_right_1d, zero_1d}, 0);

            auto pad_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
            auto pad_op = std::make_shared<ov::op::v1::Pad>(result,
                                                            pads_begin_dyn,
                                                            pads_end_dyn,
                                                            pad_value,
                                                            ov::op::PadMode::CONSTANT);
            result = pad_op;
            decomp_nodes.push_back(pad_op);
        }

        // Step 3: Reshape weights from 3D to 4D and transpose
        // Original weights: [C_out, C_in, K] -> [C_out, C_in, K, 1]
        // Mark weights for caching to avoid repeated transformation
        ov::Output<ov::Node> weights_4d;
        if (weights_shape.is_dynamic()) {
            // Get dynamic shape of weights and append 1
            auto shape_of_weights = std::make_shared<ov::op::v3::ShapeOf>(weights, ov::element::i64);
            auto weights_4d_shape =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector{shape_of_weights, one_const}, 0);
            weights_4d = std::make_shared<ov::op::v1::Reshape>(weights, weights_4d_shape, false);
        } else {
            std::vector<int64_t> weights_4d_shape = {static_cast<int64_t>(C_in_weights.get_length()),
                                                     static_cast<int64_t>(C_out.get_length()),
                                                     static_cast<int64_t>(K.get_length()),
                                                     1};
            auto weights_4d_pattern = ov::op::v0::Constant::create(ov::element::i64, {4}, weights_4d_shape);
            weights_4d = std::make_shared<ov::op::v1::Reshape>(weights, weights_4d_pattern, false);
        }

        // Transpose weights from [C_in, C_out, K, 1] to [C_out, C_in, K, 1]
        // This is needed because ConvolutionBackpropData uses [C_in, C_out, K] but
        // regular Convolution uses [C_out, C_in, K]
        std::vector<int64_t> transpose_order = {1, 0, 2, 3};
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i64, {4}, transpose_order);
        auto weights_transposed = std::make_shared<ov::op::v1::Transpose>(weights_4d, transpose_const);
        decomp_nodes.push_back(weights_4d.get_node_shared_ptr());
        decomp_nodes.push_back(weights_transposed);

        // Step 4: Regular 2D convolution with stride=1, padding=0
        ov::Strides conv_strides = {1, 1};            // Always stride=1 after upsampling
        ov::CoordinateDiff conv_pads_begin = {0, 0};  // All padding already applied
        ov::CoordinateDiff conv_pads_end = {0, 0};
        ov::Strides conv_dilations = {dilation, 1};

        auto conv = std::make_shared<ov::op::v1::Convolution>(result,
                                                              weights_transposed,
                                                              conv_strides,
                                                              conv_pads_begin,
                                                              conv_pads_end,
                                                              conv_dilations,
                                                              auto_pad);

        result = conv;
        decomp_nodes.push_back(conv);

        // Step 5: Reshape output back to 3D: [N, C_out, L_out, 1] -> [N, C_out, L_out]
        auto conv_output_shape = conv->get_output_partial_shape(0);

        // Always reshape back to 3D (remove the last dimension)
        ov::Output<ov::Node> output_3d;
        if (conv_output_shape.is_static()) {
            // Static shape case
            auto output_dims = conv_output_shape.to_shape();
            std::vector<int64_t> output_3d_shape = {static_cast<int64_t>(output_dims[0]),
                                                    static_cast<int64_t>(output_dims[1]),
                                                    static_cast<int64_t>(output_dims[2])};
            auto output_3d_pattern = ov::op::v0::Constant::create(ov::element::i64, {3}, output_3d_shape);
            output_3d = std::make_shared<ov::op::v1::Reshape>(result, output_3d_pattern, false);
        } else {
            // Dynamic shape case - get first 3 dimensions from conv output shape
            auto conv_shape = std::make_shared<ov::op::v3::ShapeOf>(result, ov::element::i64);

            // Slice to get first 3 dimensions [N, C_out, L_out]
            auto output_3d_shape = std::make_shared<ov::op::v8::Slice>(conv_shape, zero_1d, three_1d, one_const);

            output_3d = std::make_shared<ov::op::v1::Reshape>(result, output_3d_shape, false);
        }

        result = output_3d;
        decomp_nodes.push_back(output_3d.get_node_shared_ptr());

        result.get_node()->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, decomp_nodes);
        ov::replace_node(deconv, result.get_node_shared_ptr());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconv_3d, "Deconv3DDecomposition");
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
