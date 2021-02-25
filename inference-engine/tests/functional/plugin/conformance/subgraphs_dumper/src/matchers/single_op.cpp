// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op.hpp"
#include "ngraph/ops.hpp"
#include <cstdlib>

using namespace SubgraphsDumper;

template<typename dType>
bool compare_constants_data(const std::shared_ptr<ngraph::op::Constant> &op,
                            const std::shared_ptr<ngraph::op::Constant> &ref) {
    size_t elements_count = ngraph::shape_size(op->get_shape());
    if (elements_count != ngraph::shape_size(ref->get_shape())) {
        return false;
    }
    const auto &op_data = op->cast_vector<dType>();
    const auto &ref_data = ref->cast_vector<dType>();
    for (size_t i = 0; i < elements_count; ++i) {
        // std:abs doesn't implemented for unsigned types, compare explicitly to keep code universal for all dTypes
        dType diff = op_data[i] - ref_data[i] > 0 ? op_data[i] - ref_data[i] : ref_data[i] - op_data[i];
        if (diff > std::numeric_limits<dType>::epsilon()) {
            return false;
        }
    }
    return true;
}

// TODO: Move to some utils?
bool compare_constants_data(const std::shared_ptr<ngraph::op::Constant> &op,
                            const std::shared_ptr<ngraph::op::Constant> &ref) {
    switch (op->get_element_type()) {
        case ngraph::element::Type_t::boolean:
            return compare_constants_data<bool>(op, ref);
        case ngraph::element::Type_t::bf16:
            return compare_constants_data<ngraph::bfloat16>(op, ref);
        case ngraph::element::Type_t::f16:
            return compare_constants_data<ngraph::float16>(op, ref);
        case ngraph::element::Type_t::f32:
            return compare_constants_data<float>(op, ref);
        case ngraph::element::Type_t::f64:
            return compare_constants_data<double>(op, ref);
        case ngraph::element::Type_t::i8:
            return compare_constants_data<int8_t>(op, ref);
        case ngraph::element::Type_t::i16:
            return compare_constants_data<int16_t>(op, ref);
        case ngraph::element::Type_t::i32:
            return compare_constants_data<int32_t>(op, ref);
        case ngraph::element::Type_t::i64:
            return compare_constants_data<int64_t>(op, ref);
        case ngraph::element::Type_t::u8:
            return compare_constants_data<uint8_t>(op, ref);
        case ngraph::element::Type_t::u16:
            return compare_constants_data<uint16_t>(op, ref);
        case ngraph::element::Type_t::u32:
            return compare_constants_data<uint32_t>(op, ref);
        case ngraph::element::Type_t::u64:
            return compare_constants_data<uint64_t>(op, ref);
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
                    ngraph::op::v1::Convolution,
                    ngraph::op::v1::ConvolutionBackpropData,
                    ngraph::op::v1::GroupConvolution,
                    ngraph::op::v1::GroupConvolutionBackpropData,
                    ngraph::op::v0::MatMul,
                    ngraph::op::v1::Add,
                    ngraph::op::v1::Multiply,
                    ngraph::op::v1::Subtract,
                    ngraph::op::v1::Power,
                    ngraph::op::v1::ReduceMax,
                    ngraph::op::v1::ReduceMin>>(std::vector<std::string>{}, std::vector<size_t>{0, 1}),
    };
}
