// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op.hpp"
#include "ngraph/ops.hpp"

using namespace SubgraphsDumper;

// TODO: Move to some utils?
bool compare_constants_data(const std::shared_ptr<ngraph::op::Constant> &op,
                            const std::shared_ptr<ngraph::op::Constant> &ref) {
    switch (op->get_element_type()) {
        case ngraph::element::Type_t::boolean:
            if (op->cast_vector<bool>() != ref->cast_vector<bool>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::bf16:
            if (op->cast_vector<ngraph::bfloat16>() != ref->cast_vector<ngraph::bfloat16>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::f16:
            if (op->cast_vector<ngraph::float16>() != ref->cast_vector<ngraph::float16>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::f32:
            if (op->cast_vector<float>() != ref->cast_vector<float>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::f64:
            if (op->cast_vector<double>() != ref->cast_vector<double>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i8:
            if (op->cast_vector<int8_t>() != ref->cast_vector<int8_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i16:
            if (op->cast_vector<int16_t>() != ref->cast_vector<int16_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i32:
            if (op->cast_vector<int32_t>() != ref->cast_vector<int32_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::i64:
            if (op->cast_vector<int64_t>() != ref->cast_vector<int64_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u8:
            if (op->cast_vector<uint8_t>() != ref->cast_vector<uint8_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u16:
            if (op->cast_vector<uint16_t>() != ref->cast_vector<uint16_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u32:
            if (op->cast_vector<uint32_t>() != ref->cast_vector<uint32_t>()) {
                return false;
            } else {
                return true;
            }
        case ngraph::element::Type_t::u64:
            if (op->cast_vector<uint64_t>() != ref->cast_vector<uint64_t>()) {
                return false;
            } else {
                return true;
            }
        default:
            throw std::runtime_error("unsupported type");
    }
}

const char *SingleOpMatcher::name = "generic_single_op";

bool SingleOpMatcher::match(const std::shared_ptr<ngraph::Node> &node, const std::shared_ptr<ngraph::Node> &ref) {
    // Match OP type and version
    if (node->get_type_info().name != ref->get_type_info().name ||
        node->get_type_info().version != ref->get_type_info().version) {
        return false;
    }
    // Match inputs size
    if (node->get_input_size() == ref->get_input_size()) {
        // Match input ranks, element types and static shapes
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            bool rankIsEqual = node->get_input_tensor(i).get_partial_shape().rank() ==
                               ref->get_input_tensor(i).get_partial_shape().rank();
            bool elemTypeIsEqual = node->get_input_tensor(i).get_element_type() ==
                                   ref->get_input_tensor(i).get_element_type();
            bool is_dynamic = node->get_input_node_ptr(i)->is_dynamic() ==
                              ref->get_input_node_ptr(i)->is_dynamic();
            if (!(rankIsEqual && elemTypeIsEqual && is_dynamic)) {
                return false;
            }
        }
    } else {
        return false;
    }

    // Match outputs size
    if (node->get_output_size() == ref->get_output_size()) {
        // Match output element type
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            if (node->get_output_tensor(i).get_element_type() !=
                ref->get_output_tensor(i).get_element_type()) {
                return false;
            }
        }
    } else {
        return false;
    }
    // Compare node attributes
    CompareNodesAttributes attrs_comparator;
    auto &node_visitor = attrs_comparator.get_cmp_reader();
    auto &ref_visitor = attrs_comparator.get_ref_reder();
    // TODO: Need to apply readers wise-versa, otherwise comparison fails, why?
    node->visit_attributes(ref_visitor);
    ref->visit_attributes(node_visitor);
    // TODO: Implement equal() with attributes mask - which attributes to compare and which not
    if (!attrs_comparator.equal()) {
        return false;
    }
    // Match ports values
    const auto &cfg = get_config(node);
    std::vector<size_t> ignored_ports = cfg->ignored_ports;

    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        if (std::find(ignored_ports.begin(), ignored_ports.end(), port_id) != ignored_ports.end()) {
            continue;
        }
        const auto &cur_node_input = node->input_value(port_id).get_node_shared_ptr();
        const auto &ref_node_input = ref->input_value(port_id).get_node_shared_ptr();

        const auto &cur_const_input = std::dynamic_pointer_cast<ngraph::op::Constant>(cur_node_input);
        const auto &ref_const_input = std::dynamic_pointer_cast<ngraph::op::Constant>(ref_node_input);

        // Check that both OP an reference port inputs are constant and have same data
        if (cur_const_input != nullptr && ref_const_input != nullptr &&
            !compare_constants_data(cur_const_input, ref_const_input)) {
            return false;
            // Check that input nodes on the port both not constants
        } else if ((cur_const_input != nullptr && ref_const_input == nullptr) ||
                   (cur_const_input == nullptr && ref_const_input != nullptr)) {
            return false;
        }
    }
    return true;
}

SingleOpMatcher::SingleOpMatcher() {
    default_configs = {
            std::make_shared<MatcherConfig<>>(std::vector<std::string>{}, std::vector<size_t>{0}),
            std::make_shared<MatcherConfig<ngraph::opset6::FakeQuantize>>(std::vector<std::string>{},
                                                                          std::vector<size_t>{0, 1, 2, 3, 4}),
            std::make_shared<MatcherConfig<
                    ngraph::opset6::Convolution,
                    ngraph::opset6::ConvolutionBackpropData,
                    ngraph::opset6::GroupConvolution,
                    ngraph::opset6::GroupConvolutionBackpropData,
                    ngraph::opset6::MatMul,
                    ngraph::opset6::Add,
                    ngraph::opset6::Multiply,
                    ngraph::opset6::Subtract,
                    ngraph::opset6::Power,
                    ngraph::opset6::ReduceMax,
                    ngraph::opset6::ReduceMin>>(std::vector<std::string>{}, std::vector<size_t>{0, 1}),
    };
}
