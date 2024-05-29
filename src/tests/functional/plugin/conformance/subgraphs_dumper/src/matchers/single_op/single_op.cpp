// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ops.hpp"
#include "matchers/single_op/single_op.hpp"
#include "utils/node.hpp"
#include "utils/attribute_visitor.hpp"
#include "utils/model_comparator.hpp"

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

void SingleOpMatcher::set_strict_shape_match(bool strict_shape_match) {
    is_strict_shape_match = strict_shape_match;
}

void SingleOpMatcher::set_match_attrib(bool match_attrib) {
    is_match_attributes = match_attrib;
}

void SingleOpMatcher::set_match_in_types(bool match_in_types) {
    is_match_in_types = match_in_types;
}

bool SingleOpMatcher::match_inputs(const std::shared_ptr<ov::Node> &node,
                                   const std::shared_ptr<ov::Node> &ref) const {
    if (node->get_input_size() != ref->get_input_size()) {
        return false;
    }
    const auto &cfg = get_config(node);
    const std::vector<size_t> &ignored_ports = cfg->ignored_ports;

    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        if (std::find(ignored_ports.begin(), ignored_ports.end(), port_id) != ignored_ports.end()) {
            continue;
        }
        if (node->get_input_element_type(port_id) != ref->get_input_element_type(port_id)) {
            return false;
        }
        const auto& partial_shape = node->get_input_partial_shape(port_id);
        const auto& ref_partial_shape = ref->get_input_partial_shape(port_id);
        if (is_strict_shape_match && partial_shape != ref_partial_shape) {
            return false;
        } else if (partial_shape.rank() != ref_partial_shape.rank()) {
            return false;
        }
        if (partial_shape.is_dynamic() != ref_partial_shape.is_dynamic()) {
            return false;
        }
        if (is_match_in_types) {
            const auto& in_node = node->get_input_node_shared_ptr(port_id);
            const auto& in_node_ref = ref->get_input_node_shared_ptr(port_id);
            if (ov::util::is_node_to_skip(in_node) || ov::util::is_node_to_skip(in_node_ref)) {
                continue;
            } else if (in_node->get_type_info() != in_node_ref->get_type_info()) {
                return false;
            }
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
        if (node->get_output_element_type(port_id) != ref->get_output_element_type(port_id)) {
            return false;
        }

        const auto& partial_shape = node->get_output_partial_shape(port_id);
        const auto& ref_partial_shape = ref->get_output_partial_shape(port_id);
        if (partial_shape.is_dynamic() != ref_partial_shape.is_dynamic()) {
            return false;
        }
        if (is_strict_shape_match && partial_shape != ref_partial_shape) {
            return false;
        } else if (partial_shape.rank() != ref_partial_shape.rank()) {
            return false;
        }
    }
    return true;
}

bool SingleOpMatcher::match_attrs(const std::shared_ptr<ov::Node> &node,
                                  const std::shared_ptr<ov::Node> &ref) const {
    util::ReadAttributes visitor_node, visitor_ref;
    node->visit_attributes(visitor_node);
    ref->visit_attributes(visitor_ref);

    {
        auto node_bodies = visitor_node.get_model_attributes_map();
        auto ref_node_bodies = visitor_ref.get_model_attributes_map();
        bool is_match = true;

        auto model_comparator = ov::util::ModelComparator::get();
        auto match_coefficient = model_comparator->get_match_coefficient();
        model_comparator->set_match_coefficient(1.f);
        for (const auto& body : node_bodies) {
            if (!ref_node_bodies.count(body.first)) {
                is_match = false;
                break;
            }
            if (!model_comparator->match(body.second, ref_node_bodies.at(body.first))) {
                is_match = false;
                break;
            }
        }
        model_comparator->set_match_coefficient(match_coefficient);
        if (!is_match) {
            return false;
        }
    }
    return visitor_node.get_attributes_map() == visitor_ref.get_attributes_map();
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
    if (is_match_attributes) {
        if (!match_attrs(node, ref) && !ov::util::is_node_to_skip(node)) {
            return false;
        }
    }
    if (!match_inputs(node, ref)) {
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
            std::make_shared<MatcherConfig<
                    ov::op::v1::Convolution,
                    ov::op::v1::ConvolutionBackpropData,
                    ov::op::v1::GroupConvolution,
                    ov::op::v1::GroupConvolutionBackpropData>>(true)
    };
}
