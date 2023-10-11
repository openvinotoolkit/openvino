// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/util/op_types.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "matchers/single_op/single_op.hpp"

using namespace ov::tools::subgraph_dumper;

iMatcherConfig::Ptr SingleOpMatcher::get_config(const std::shared_ptr<ov::Node> &node) const {
    for (const auto &cfg : default_configs) {
        if (cfg->op_in_config(node)) {
            return cfg;
        }
    }
    for (const auto &cfg : default_configs) {
        if (cfg->is_fallback_config) {
            return cfg;
        }
    }
    return std::make_shared<MatcherConfig<>>();
}

bool SingleOpMatcher::match_inputs(const std::shared_ptr<ov::Node> &node,
                                   const std::shared_ptr<ov::Node> &ref) const {
    if (node->get_input_size() != ref->get_input_size()) {
        return false;
    }
    const std::vector<size_t> &ignored_ports = get_config(node)->ignored_ports;

    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        if (std::find(ignored_ports.begin(), ignored_ports.end(), port_id) != ignored_ports.end()) {
            continue;
        }
        if (!ov::op::util::is_parameter(node) && !ov::op::util::is_parameter(ref) &&
            !ov::op::util::is_constant(node) && !ov::op::util::is_constant(ref)) {
            const auto &cur_node_input_type = node->input_value(port_id).get_node_shared_ptr()->get_type_info();
            const auto &ref_node_input_type = ref->input_value(port_id).get_node_shared_ptr()->get_type_info();
            if (cur_node_input_type != ref_node_input_type) {
                return false;
            }
        }
        if (node->get_input_tensor(port_id).get_partial_shape().rank() != ref->get_input_tensor(port_id).get_partial_shape().rank()) {
            return false;
        }
        if (node->get_input_tensor(port_id).get_element_type() != ref->get_input_tensor(port_id).get_element_type()) {
            return false;
        }
        if (node->get_input_partial_shape(port_id).is_dynamic() != ref->get_input_partial_shape(port_id).is_dynamic()) {
            return false;
        }
    }
    return true;
}

bool
SingleOpMatcher::match_outputs(const std::shared_ptr<ov::Node> &node,
                               const std::shared_ptr<ov::Node> &ref) const {
    if (node->get_output_size() != ref->get_output_size()) {
        return false;
    }
    for (size_t port_id = 0; port_id < node->get_output_size(); ++port_id) {
        if (!ov::op::util::is_output(node) && !ov::op::util::is_output(ref)) {
            const auto &cur_node_out_type = node->output(port_id).get_node_shared_ptr()->get_type_info();
            const auto &ref_node_out_type = ref->output(port_id).get_node_shared_ptr()->get_type_info();
            if (cur_node_out_type != ref_node_out_type) {
                return false;
            }
        }
        if (node->get_output_tensor(port_id).get_element_type() != ref->get_output_tensor(port_id).get_element_type()) {
            return false;
        }
        if (node->get_output_tensor(port_id).get_partial_shape().is_dynamic() != ref->get_output_tensor(port_id).get_partial_shape().is_dynamic()) {
            return false;
        }
        if (node->get_output_tensor(port_id).get_partial_shape().rank()!= ref->get_output_tensor(port_id).get_partial_shape().rank()) {
            return false;
        }
    }
    return true;
}

bool SingleOpMatcher::match_attrs(const std::shared_ptr<ov::Node> &node,
                                  const std::shared_ptr<ov::Node> &ref) const {
    // todo: iefode: to provide correct with ingored attributes
    return attributes::compare(node.get(), ref.get(), Comparator::CmpValues::ATTRIBUTES).valid;
}

bool SingleOpMatcher::match(const std::shared_ptr<ov::Node> &node,
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
    if (!match_attrs(node, ref) && !ov::op::util::is_parameter(node) && !ov::op::util::is_parameter(ref)) {
        return false;
    }
    if (!match_outputs(node, ref)) {
        return false;
    }
    return true;
}

bool SingleOpMatcher::same_op_type(const std::shared_ptr<ov::Node> &node,
                                   const std::shared_ptr<ov::Node> &ref) const {
    return node->get_type_info() == ref->get_type_info();
}

SingleOpMatcher::SingleOpMatcher() {
    default_configs = {
            // std::make_shared<MatcherConfig<>>(std::vector<std::string>{}, std::vector<size_t>{0}),
            // std::make_shared<MatcherConfig<ov::opset8::FakeQuantize>>(std::vector<std::string>{},
            //                                                           std::vector<size_t>{0, 1, 2, 3, 4}),
            std::make_shared<MatcherConfig<
                    ov::op::v1::Convolution,
                    ov::op::v1::ConvolutionBackpropData,
                    ov::op::v1::GroupConvolution,
                    ov::op::v1::GroupConvolutionBackpropData>>(true)
    };
}
