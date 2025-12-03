// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convolution_bias_fusion.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/convolution.hpp"
#include "transformations/utils/utils.hpp"

static inline std::vector<size_t> getNormalizedDimsBySize(const std::vector<size_t>& dims, size_t ndims) {
    auto normalizedDims = dims;
    if (size_t num_missing_dims = ndims - dims.size(); num_missing_dims <= ndims) {
        normalizedDims.insert(normalizedDims.begin(), num_missing_dims, 1);
    }
    return normalizedDims;
}

ov::pass::ConvolutionBiasFusion::ConvolutionBiasFusion() {
    MATCHER_SCOPE(ConvolutionBiasFusion);
    using namespace ov::pass::pattern;

    auto data_batch = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto filters = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto m_conv = ov::pass::pattern::wrap_type<ov::op::internal::Convolution>({data_batch, filters},
                                                                              ov::pass::pattern::consumers_count(1));
    auto m_bias = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_conv, m_bias});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();

        auto conv = ov::as_type_ptr<ov::op::internal::Convolution>(pattern_to_output[m_conv].get_node_shared_ptr());
        if (!conv || transformation_callback(conv)) {
            return false;
        }

        const auto& bias = pattern_to_output[m_bias].get_node_shared_ptr();
        if (!bias || transformation_callback(bias)) {
            return false;
        }

        auto add = ov::as_type_ptr<ov::op::v1::Add>(pattern_to_output[m_add].get_node_shared_ptr());
        if (!add || transformation_callback(add)) {
            return false;
        }

        // Check if the bias node is suitable for fusion
        auto isSuitableChildNode = [&](const std::shared_ptr<ov::Node>& parentNode,
                                       const std::shared_ptr<ov::Node>& childNode) {
            if (childNode->get_type_info() != ov::op::v1::Add::get_type_info_static() ||
                childNode->get_input_size() != 2) {
                return false;
            }

            // Determine which input is the bias (should be the one that's not the convolution)
            auto biasPort = childNode->get_input_node_shared_ptr(0) == parentNode ? 1 : 0;
            const auto biasNode = childNode->get_input_node_shared_ptr(biasPort);

            // Check if bias node is a constant
            if (!std::dynamic_pointer_cast<ov::op::v0::Constant>(biasNode) || biasNode->get_output_size() != 1) {
                return false;
            }

            const auto parentOutDims = parentNode->get_output_partial_shape(0);
            if (parentOutDims.rank().is_dynamic()) {
                return false;
            }

            const auto biasDims = biasNode->get_output_partial_shape(0);
            if (biasDims.rank().is_dynamic()) {
                return false;
            }

            auto rank = parentOutDims.size();
            const auto norm_bias = getNormalizedDimsBySize(biasNode->get_output_shape(0), rank);

            if (parentOutDims.size() != norm_bias.size() || norm_bias.size() < 2) {
                return false;
            }

            const auto channelAxis = 1;  // Channel axis for convolution output
            if (channelAxis >= static_cast<int>(parentOutDims.size())) {
                return false;
            }

            // Check if bias matches the channel dimension
            if (!parentOutDims[channelAxis].is_static() ||
                norm_bias[channelAxis] != static_cast<size_t>(parentOutDims[channelAxis].get_length())) {
                return false;
            }

            // Check that all other dimensions are 1 (broadcasting requirement)
            for (size_t i = 0; i < norm_bias.size(); i++) {
                if (norm_bias[i] != 1 && static_cast<int>(i) != channelAxis) {
                    return false;
                }
            }

            return true;
        };

        if (!isSuitableChildNode(conv, add)) {
            return false;
        }

        const ov::PartialShape& output_shape = conv->get_output_partial_shape(0);
        auto rank = output_shape.size();
        if (rank == 0) {
            return false;
        }
        ov::NodeVector new_ops;

        std::shared_ptr<ov::Node> final_bias = bias;
        auto add_shape = add->get_output_partial_shape(0);

        if (add_shape.rank().is_dynamic()) {
            return false;
        }

        if (add_shape.size() >= 2) {
            auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
            final_bias = ov::op::util::make_try_fold<ov::op::v1::Reshape>(final_bias, reshape_const, true);
            new_ops.push_back(final_bias);
        }

        auto new_conv = std::make_shared<ov::op::internal::Convolution>(conv->input_value(0),
                                                                        conv->input_value(1),
                                                                        final_bias,
                                                                        conv->get_strides(),
                                                                        conv->get_pads_begin(),
                                                                        conv->get_pads_end(),
                                                                        conv->get_dilations(),
                                                                        conv->get_groups(),
                                                                        conv->get_auto_pad(),
                                                                        conv->get_output_element_type(0));

        new_ops.push_back(new_conv);

        new_conv->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({conv, add}, new_ops);
        ov::replace_node(add, new_conv);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_add, matcher_name);
    this->register_matcher(m, callback);
}
