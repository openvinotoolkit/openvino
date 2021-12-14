// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/convolutions.hpp"

using namespace SubgraphsDumper;
ConvolutionsMatcher::ConvolutionsMatcher() {
    default_configs = {
            std::make_shared<MatcherConfig<
                    ngraph::op::v1::Convolution,
                    ngraph::op::v1::ConvolutionBackpropData,
                    ngraph::op::v1::GroupConvolution,
                    ngraph::op::v1::GroupConvolutionBackpropData>>(std::vector<std::string>{}, std::vector<size_t>{0, 1})
    };
}

bool ConvolutionsMatcher::match(const std::shared_ptr<ngraph::Node> &node,
                            const std::shared_ptr<ngraph::Node> &ref,
                            const LayerTestsUtils::OPInfo &op_info) const {
    const auto &cfg = get_config(node);
    if (match_only_configured_ops() && cfg->is_fallback_config) {
        return false;
    }
    if (cfg->ignore_matching) {
        return false;
    }
    return same_op_type(node, ref, op_info) &&
           match_inputs(node, ref, op_info) &&
           match_outputs(node, ref, op_info) &&
           same_attrs(node, ref, op_info) &&
           match_ports(node, ref, op_info);
}
bool ConvolutionsMatcher::match_inputs(const std::shared_ptr<ngraph::Node> &node,
                                       const std::shared_ptr<ngraph::Node> &ref,
                                       const LayerTestsUtils::OPInfo &op_info) const {
    if (node->get_input_size() != ref->get_input_size()) {
        return false;
    }
    bool rankIsEqual = node->get_input_tensor(0).get_partial_shape().rank() ==
                       ref->get_input_tensor(0).get_partial_shape().rank();
    bool elemTypeIsEqual = node->get_input_tensor(0).get_element_type() ==
                           ref->get_input_tensor(0).get_element_type();
    bool is_dynamic = node->get_input_node_ptr(0)->is_dynamic() ==
                      ref->get_input_node_ptr(0)->is_dynamic();
    if (!(rankIsEqual && elemTypeIsEqual && is_dynamic)) {
        return false;
    }
    bool has_groups = std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolution>(node) != nullptr ||
                      std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolutionBackpropData>(node);
    size_t kernel_size_offset = has_groups ? 3 : 2;
    auto ref_weights_shape = ref->get_input_tensor(1).get_shape();
    auto cur_weights_shape = node->get_input_tensor(1).get_shape();
    const auto ref_kernel_size = std::vector<size_t>(ref_weights_shape.begin() + kernel_size_offset,
                                                     ref_weights_shape.end());
    const auto cur_kernel_size = std::vector<size_t>(cur_weights_shape.begin() + kernel_size_offset,
                                                     cur_weights_shape.end());
    if (ref_kernel_size != cur_kernel_size) {
        return false;
    }
    return true;
}
