// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rotate_inputs.hpp"

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ops/gna_convolution.hpp"

using namespace ov::opset11;
using namespace ov::pass;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::graph_utils;

namespace {
inline bool is_skip_operation(const std::shared_ptr<ov::Node>& node) {
    return (std::dynamic_pointer_cast<Reshape>(node) != nullptr ||
            std::dynamic_pointer_cast<Transpose>(node) != nullptr ||
            std::dynamic_pointer_cast<Squeeze>(node) != nullptr ||
            std::dynamic_pointer_cast<Unsqueeze>(node) != nullptr ||
            std::dynamic_pointer_cast<FakeQuantize>(node) != nullptr) &&
           has_n_consumers(node, 1);
}
}  // namespace

InsertConvolutionTransposeHW::InsertConvolutionTransposeHW() {
    MATCHER_SCOPE(InsertConvolutionTransposeHW);

    const auto conv_pattern = pattern::wrap_type<op::GNAConvolution>(
        {pattern::any_input(), pattern::any_input()},
        [](const ov::Output<ov::Node>& node) {
            std::shared_ptr<op::GNAConvolution> conv =
                std::dynamic_pointer_cast<op::GNAConvolution>(node.get_node_shared_ptr());
            helper::ConvData conv_data;
            helper::GetConvData(conv, conv_data);
            auto validator = limitations::Limitations::get_instance()->get_cnn_validator();
            return (validator && !validator->ShouldUseOnlyConv2DGnaIface()) &&
                   gna_convolution_layer::isMappableFrom2DTo1D(static_cast<uint32_t>(conv_data.input_height),
                                                               static_cast<uint32_t>(conv_data.input_width),
                                                               static_cast<uint32_t>(conv_data.input_channel_count),
                                                               static_cast<uint32_t>(conv_data.filter_height),
                                                               static_cast<uint32_t>(conv_data.filter_width),
                                                               static_cast<uint32_t>(conv_data.filter_stride_height),
                                                               static_cast<uint32_t>(conv_data.filter_stride_width)) &&
                   gna_convolution_layer::should_transpose_h_w(static_cast<uint32_t>(conv_data.input_height),
                                                               static_cast<uint32_t>(conv_data.filter_height),
                                                               static_cast<uint32_t>(conv_data.input_channel_count),
                                                               static_cast<uint32_t>(conv_data.filter_stride_height));
        });

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        // auto param_node = pattern_map.at(param_pattern).get_node_shared_ptr();
        auto conv_node = pattern_map.at(conv_pattern).get_node_shared_ptr();

        std::shared_ptr<ov::Node> target_node =
            graph_utils::get_prev_node_skipping_certain(conv_node->get_input_node_shared_ptr(0), is_skip_operation);
        std::shared_ptr<Parameter> param_node = std::dynamic_pointer_cast<Parameter>(target_node);

        if (!param_node) {
            return false;
        }

        // transpose all convolution inputs
        for (const auto& conv_input : conv_node->inputs()) {
            // Transpose H and W (NHWC -> NWHC)
            ov::AxisVector tr_axis = {0, 2, 1, 3};
            auto transpose_const = std::make_shared<Constant>(ov::element::i8, ov::Shape{tr_axis.size()}, tr_axis);
            auto transpose = std::make_shared<Transpose>(conv_input.get_source_output(), transpose_const);

            // Reshape out
            ov::Shape shape_out = conv_input.get_shape();
            auto reshape_out_const =
                std::make_shared<Constant>(ov::element::i32, ov::Shape{shape_out.size()}, shape_out);
            auto reshape_out = std::make_shared<Reshape>(transpose, reshape_out_const, false);

            conv_input.replace_source_output(reshape_out);
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}
