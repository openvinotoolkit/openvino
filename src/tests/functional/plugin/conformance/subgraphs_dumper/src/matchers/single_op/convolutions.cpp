// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "matchers/single_op/convolutions.hpp"

using namespace ov::tools::subgraph_dumper;

ConvolutionsMatcher::ConvolutionsMatcher() {
    default_configs = {
            std::make_shared<MatcherConfig<
                    ov::op::v1::Convolution,
                    ov::op::v1::ConvolutionBackpropData,
                    ov::op::v1::GroupConvolution,
                    ov::op::v1::GroupConvolutionBackpropData>>(std::vector<std::string>{}, std::vector<size_t>{1})
    };
}

bool ConvolutionsMatcher::match(const std::shared_ptr<ov::Node> &node,
                                const std::shared_ptr<ov::Node> &ref) const {
    const auto &cfg = get_config(node);
    if (match_only_configured_ops() && cfg->is_fallback_config) {
        return false;
    }
    if (cfg->ignore_matching) {
        return false;
    }
    if (!same_op_type(node, ref)) {
        return false;
    }
    if (!match_inputs(node, ref)) {
        return false;
    }
    if (!match_attrs(node, ref)) {
        return false;
    }
    if (!match_outputs(node, ref)) {
        return false;
    }
    return true;
}

bool ConvolutionsMatcher::match_inputs(const std::shared_ptr<ov::Node> &node,
                                       const std::shared_ptr<ov::Node> &ref) const {
    if (!SingleOpMatcher::match_inputs(node, ref)) {
        return false;
    }
    bool has_groups = ov::as_type_ptr<ov::op::v1::GroupConvolution>(node) ||
                      ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node);
    size_t kernel_size_offset = has_groups ? 3 : 2;
    auto ref_weights_shape = ref->get_input_partial_shape(1).get_shape();
    auto cur_weights_shape = node->get_input_partial_shape(1).get_shape();
    if (is_strict_shape_match && ref_weights_shape != cur_weights_shape) {
        return false;
    }
    const auto ref_kernel_size = std::vector<size_t>(ref_weights_shape.begin() + kernel_size_offset,
                                                     ref_weights_shape.end());
    const auto cur_kernel_size = std::vector<size_t>(cur_weights_shape.begin() + kernel_size_offset,
                                                     cur_weights_shape.end());
    if (ref_kernel_size != cur_kernel_size) {
        return false;
    }
    return true;
}
