// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconv_1d_decomposition.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

Deconv1DDecomposition::Deconv1DDecomposition() {
    auto input_pattern = ov::pass::pattern::any_input();
    auto weights_pattern = ov::pass::pattern::any_input();
    auto deconv_1d =
        ov::pass::pattern::wrap_type<ov::op::v1::ConvolutionBackpropData>({input_pattern, weights_pattern},
                                                                          ov::pass::pattern::consumers_count(1));

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto deconv = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(node);
        if (!deconv) {
            return false;
        }

        auto input_shape = deconv->get_input_partial_shape(0);
        auto weights_shape = deconv->get_input_partial_shape(1);

        // Only apply to 1D deconv (3D tensors)
        if (!input_shape.rank().is_static() || input_shape.rank().get_length() != 3 ||
            !weights_shape.rank().is_static() || weights_shape.rank().get_length() != 3) {
            return false;
        }

        auto input = deconv->input_value(0);
        auto weights = deconv->input_value(1);

        auto strides = deconv->get_strides();
        auto pads_begin = deconv->get_pads_begin();
        auto pads_end = deconv->get_pads_end();
        auto dilations = deconv->get_dilations();
        auto auto_pad = deconv->get_auto_pad();

        bool is_auto_pad = (auto_pad != ov::op::PadType::EXPLICIT && auto_pad != ov::op::PadType::NOTSET);
        bool can_calculate_padding = !input_shape.is_dynamic() && !weights_shape.is_dynamic();

        if (is_auto_pad && can_calculate_padding) {
            auto stride = strides[0];

            if (auto_pad == ov::op::PadType::SAME_UPPER || auto_pad == ov::op::PadType::SAME_LOWER) {
                auto total_padding = std::max<int64_t>(0, stride - 1);

                if (auto_pad == ov::op::PadType::SAME_UPPER) {
                    pads_begin[0] = total_padding / 2;
                    pads_end[0] = total_padding - pads_begin[0];
                } else {
                    pads_end[0] = total_padding / 2;
                    pads_begin[0] = total_padding - pads_end[0];
                }
            } else if (auto_pad == ov::op::PadType::VALID) {
                pads_begin[0] = 0;
                pads_end[0] = 0;
            }

            auto_pad = ov::op::PadType::EXPLICIT;
        }
        auto output_padding = deconv->get_output_padding();

        if (strides.size() != 1 || pads_begin.size() != 1 || pads_end.size() != 1 || dilations.size() != 1) {
            return false;
        }
        auto stride = strides[0];
        auto pad_begin = pads_begin[0];
        auto pad_end = pads_end[0];
        auto dilation = dilations[0];
        auto out_pad = output_padding.empty() ? 0 : output_padding[0];

        const auto& C_in = input_shape[1];
        const auto& C_in_weights = weights_shape[0];

        if (C_in != C_in_weights) {
            return false;
        }

        auto result = input;
        std::vector<std::shared_ptr<ov::Node>> decomp_nodes;

        auto zero_scalar = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto one_scalar = ov::op::v0::Constant::create(ov::element::i32, {}, {1});
        auto two_scalar = ov::op::v0::Constant::create(ov::element::i32, {}, {2});
        auto minus_one = ov::op::v0::Constant::create(ov::element::i32, {}, {-1});

        auto one_const = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto zero_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto three_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {3});
        auto minus_one_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {-1});

        auto create_shape_pattern = [&](const std::vector<ov::Output<ov::Node>>& dims) -> ov::Output<ov::Node> {
            if (dims.size() == 1) {
                return dims[0];
            }
            return ov::op::util::make_try_fold<ov::op::v0::Concat>(dims, 0);
        };

        auto make_1d = [&](const ov::Output<ov::Node>& scalar) -> ov::Output<ov::Node> {
            return ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(scalar, zero_scalar);
        };

        std::shared_ptr<ov::Node> shape_of_weights;

        auto shape_of_input = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(input, ov::element::i32);
        decomp_nodes.push_back(shape_of_input);
        auto gather_n = ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_of_input, zero_scalar, zero_scalar);
        auto gather_c = ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_of_input, one_scalar, zero_scalar);
        auto gather_l = ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_of_input, two_scalar, zero_scalar);
        decomp_nodes.push_back(gather_n);
        decomp_nodes.push_back(gather_c);
        decomp_nodes.push_back(gather_l);

        auto N_node = make_1d(gather_n);
        auto C_in_node = make_1d(gather_c);
        auto L_node = make_1d(gather_l);
        decomp_nodes.push_back(N_node.get_node_shared_ptr());
        decomp_nodes.push_back(C_in_node.get_node_shared_ptr());
        decomp_nodes.push_back(L_node.get_node_shared_ptr());
        auto input_4d_pattern = create_shape_pattern({N_node, C_in_node, L_node, one_const});
        decomp_nodes.push_back(input_4d_pattern.get_node_shared_ptr());
        auto input_4d = std::make_shared<ov::op::v1::Reshape>(input, input_4d_pattern, false);
        result = input_4d;
        decomp_nodes.push_back(input_4d);

        if (stride > 1) {
            auto stride_const = ov::op::v0::Constant::create(ov::element::i32, {}, {stride});
            auto stride_const_1d = ov::op::v0::Constant::create(ov::element::i32, {1}, {stride});

            auto L_minus_1 = ov::op::util::make_try_fold<ov::op::v1::Add>(L_node, minus_one_1d);
            decomp_nodes.push_back(L_minus_1);
            auto L_minus_1_times_stride = ov::op::util::make_try_fold<ov::op::v1::Multiply>(L_minus_1, stride_const_1d);
            decomp_nodes.push_back(L_minus_1_times_stride);
            auto upsampled_L_node = ov::op::util::make_try_fold<ov::op::v1::Add>(L_minus_1_times_stride, one_const);
            decomp_nodes.push_back(upsampled_L_node);

            auto zero_shape = create_shape_pattern({N_node, C_in_node, upsampled_L_node, one_const});
            decomp_nodes.push_back(zero_shape.get_node_shared_ptr());
            auto zero_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
            auto zero_tensor = std::make_shared<ov::op::v3::Broadcast>(zero_value, zero_shape);
            decomp_nodes.push_back(zero_tensor);

            auto L_scalar = ov::op::util::make_try_fold<ov::op::v0::Squeeze>(L_node, zero_scalar);
            decomp_nodes.push_back(L_scalar);
            auto range =
                ov::op::util::make_try_fold<ov::op::v4::Range>(zero_scalar, L_scalar, one_scalar, ov::element::i32);
            decomp_nodes.push_back(range);
            auto indices = ov::op::util::make_try_fold<ov::op::v1::Multiply>(range, stride_const);
            decomp_nodes.push_back(indices);

            auto N_times_C_in = ov::op::util::make_try_fold<ov::op::v1::Multiply>(N_node, C_in_node);
            decomp_nodes.push_back(N_times_C_in);
            auto input_3d_shape = create_shape_pattern({N_times_C_in, L_node, one_const});
            decomp_nodes.push_back(input_3d_shape.get_node_shared_ptr());
            auto input_3d = std::make_shared<ov::op::v1::Reshape>(result, input_3d_shape, true);

            // Reshape zero tensors to [N*C_in, upsampled_L, 1] using the subgraph
            auto zero_3d_shape = create_shape_pattern({N_times_C_in, upsampled_L_node, one_const});
            decomp_nodes.push_back(zero_3d_shape.get_node_shared_ptr());
            auto zero_3d = std::make_shared<ov::op::v1::Reshape>(zero_tensor, zero_3d_shape, true);

            // Scatter update along axis 1
            auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(zero_3d, indices, input_3d, one_scalar);

            // Reshape back to 4D using subgraph
            auto upsampled_4d_shape = create_shape_pattern({N_node, C_in_node, upsampled_L_node, one_const});
            decomp_nodes.push_back(upsampled_4d_shape.get_node_shared_ptr());
            auto upsampled = std::make_shared<ov::op::v1::Reshape>(scatter, upsampled_4d_shape, false);

            result = upsampled;
            decomp_nodes.push_back(input_3d);
            decomp_nodes.push_back(zero_3d);
            decomp_nodes.push_back(scatter);
            decomp_nodes.push_back(upsampled);
        }

        {
            if (!shape_of_weights) {
                shape_of_weights = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(weights, ov::element::i32);
                decomp_nodes.push_back(shape_of_weights);
            }
            auto K_dyn = ov::op::util::make_try_fold<ov::op::v8::Gather>(shape_of_weights, two_scalar, zero_scalar);
            decomp_nodes.push_back(K_dyn);

            ov::Output<ov::Node> pad_left_dyn, pad_right_dyn;

            if (is_auto_pad && !can_calculate_padding) {
                auto K_minus_1 = ov::op::util::make_try_fold<ov::op::v1::Add>(K_dyn, minus_one);
                decomp_nodes.push_back(K_minus_1);
                auto two_const = ov::op::v0::Constant::create(ov::element::i32, {}, {2});
                auto pad_val = ov::op::util::make_try_fold<ov::op::v1::Divide>(K_minus_1, two_const, true);
                decomp_nodes.push_back(pad_val);

                if (auto_pad == ov::op::PadType::SAME_UPPER) {
                    pad_left_dyn = pad_val;  // floor((K-1)/2)
                    // pad_right = K - 1 - pad_left
                    pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_left_dyn);
                    decomp_nodes.push_back(pad_right_dyn.get_node_shared_ptr());
                } else if (auto_pad == ov::op::PadType::SAME_LOWER) {
                    // pad_left = ceil((K-1)/2) = floor(K/2)
                    pad_left_dyn = ov::op::util::make_try_fold<ov::op::v1::Divide>(K_dyn, two_const, true);
                    decomp_nodes.push_back(pad_left_dyn.get_node_shared_ptr());
                    // pad_right = K - 1 - pad_left
                    pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_left_dyn);
                    decomp_nodes.push_back(pad_right_dyn.get_node_shared_ptr());
                } else if (auto_pad == ov::op::PadType::VALID) {
                    // No padding needed
                    pad_left_dyn = zero_scalar;
                    pad_right_dyn = zero_scalar;
                }

                // Reset auto_pad to EXPLICIT since we manually applied padding
                auto_pad = ov::op::PadType::EXPLICIT;
            } else {
                // For explicit padding: pad_left = K - 1 - pad_begin, pad_right = K - 1 - pad_end + out_pad
                auto pad_begin_const =
                    ov::op::v0::Constant::create(ov::element::i32, {}, {static_cast<int32_t>(pad_begin)});
                auto pad_end_const =
                    ov::op::v0::Constant::create(ov::element::i32, {}, {static_cast<int32_t>(pad_end)});
                auto out_pad_const =
                    ov::op::v0::Constant::create(ov::element::i32, {}, {static_cast<int32_t>(out_pad)});

                auto K_minus_1 = ov::op::util::make_try_fold<ov::op::v1::Add>(K_dyn, minus_one);
                decomp_nodes.push_back(K_minus_1);

                // pad_left = K - 1 - pad_begin
                pad_left_dyn = ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_begin_const);
                decomp_nodes.push_back(pad_left_dyn.get_node_shared_ptr());

                // pad_right = K - 1 - pad_end + out_pad
                auto K_minus_1_minus_pad_end =
                    ov::op::util::make_try_fold<ov::op::v1::Subtract>(K_minus_1, pad_end_const);
                pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Add>(K_minus_1_minus_pad_end, out_pad_const);
                decomp_nodes.push_back(pad_right_dyn.get_node_shared_ptr());
            }

            // Ensure non-negative padding using Maximum operation
            auto zero_const = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
            pad_left_dyn = ov::op::util::make_try_fold<ov::op::v1::Maximum>(pad_left_dyn, zero_const);
            pad_right_dyn = ov::op::util::make_try_fold<ov::op::v1::Maximum>(pad_right_dyn, zero_const);
            decomp_nodes.push_back(pad_left_dyn.get_node_shared_ptr());
            decomp_nodes.push_back(pad_right_dyn.get_node_shared_ptr());

            // Create padding vectors [0, 0, pad_left, 0] and [0, 0, pad_right, 0]
            auto pad_left_1d = make_1d(pad_left_dyn);
            auto pad_right_1d = make_1d(pad_right_dyn);
            decomp_nodes.push_back(pad_left_1d.get_node_shared_ptr());
            decomp_nodes.push_back(pad_right_1d.get_node_shared_ptr());

            auto pads_begin_dyn = ov::op::util::make_try_fold<ov::op::v0::Concat>(
                ov::OutputVector{zero_1d, zero_1d, pad_left_1d, zero_1d},
                0);
            auto pads_end_dyn = ov::op::util::make_try_fold<ov::op::v0::Concat>(
                ov::OutputVector{zero_1d, zero_1d, pad_right_1d, zero_1d},
                0);
            decomp_nodes.push_back(pads_begin_dyn);
            decomp_nodes.push_back(pads_end_dyn);

            auto pad_value = ov::op::v0::Constant::create(result.get_element_type(), {}, {0});
            auto pad_op = std::make_shared<ov::op::v1::Pad>(result,
                                                            pads_begin_dyn,
                                                            pads_end_dyn,
                                                            pad_value,
                                                            ov::op::PadMode::CONSTANT);
            result = pad_op;
            decomp_nodes.push_back(pad_op);
        }

        if (!shape_of_weights) {
            shape_of_weights = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(weights, ov::element::i32);
            decomp_nodes.push_back(shape_of_weights);
        }
        auto weights_4d_shape =
            ov::op::util::make_try_fold<ov::op::v0::Concat>(ov::OutputVector{shape_of_weights, one_const}, 0);
        decomp_nodes.push_back(weights_4d_shape);
        ov::Output<ov::Node> weights_4d = std::make_shared<ov::op::v1::Reshape>(weights, weights_4d_shape, false);

        std::vector<int32_t> transpose_order = {1, 0, 2, 3};
        auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {4}, transpose_order);
        auto weights_transposed = std::make_shared<ov::op::v1::Transpose>(weights_4d, transpose_const);
        decomp_nodes.push_back(weights_4d.get_node_shared_ptr());
        decomp_nodes.push_back(weights_transposed);

        ov::Strides conv_strides = {1, 1};
        ov::CoordinateDiff conv_pads_begin = {0, 0};
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

        auto conv_shape = ov::op::util::make_try_fold<ov::op::v3::ShapeOf>(result, ov::element::i32);
        decomp_nodes.push_back(conv_shape);
        auto output_3d_shape = ov::op::util::make_try_fold<ov::op::v8::Slice>(conv_shape, zero_1d, three_1d, one_const);
        decomp_nodes.push_back(output_3d_shape);

        auto output_3d = std::make_shared<ov::op::v1::Reshape>(result, output_3d_shape, false);

        result = output_3d;
        decomp_nodes.push_back(output_3d);

        result.get_node()->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, decomp_nodes);
        ov::replace_node(deconv, result.get_node_shared_ptr());

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconv_1d, "Deconv1DDecomposition");
    register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
