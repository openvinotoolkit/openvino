// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_cloner.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "common_test_utils/data_utils.hpp"

namespace SubgraphsDumper {
template<>
const std::shared_ptr<ngraph::Node> clone_with_new_inputs<ngraph::Node>(const std::shared_ptr<ngraph::Node> &node) {
    ngraph::OutputVector op_inputs;
    for (const auto &input : node->inputs()) {
        if (ngraph::op::is_constant(input.get_source_output().get_node_shared_ptr())) {
            op_inputs.push_back(input.get_source_output().get_node_shared_ptr()->clone_with_new_inputs({}));
        } else {
            op_inputs.push_back(std::make_shared<ngraph::op::Parameter>(input.get_element_type(),
                                                                        input.get_source_output().get_shape()));
        }
    }
    auto op_clone = node->clone_with_new_inputs(op_inputs);
    return op_clone;
}

template<typename dType>
std::shared_ptr<ngraph::op::Constant>
copy_constant_with_randomization(const std::shared_ptr<ngraph::op::Constant> &const_node) {
    std::vector<dType> data = const_node->cast_vector<dType>();
    dType min = *std::min_element(data.begin(), data.end());
    dType max = *std::min_element(data.begin(), data.end());
    CommonTestUtils::fill_data_random<dType>(data.data(), data.size(), max, min, 1, 1);
    return std::make_shared<ngraph::op::Constant>(const_node->get_element_type(), const_node->get_shape(), data);
}

std::shared_ptr<ngraph::Node> clone_weightable_node(const std::shared_ptr<ngraph::Node> &node,
                                                    const std::vector<size_t> &weight_ports) {
    ngraph::OutputVector op_inputs;
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto input = node->input(i).get_source_output().get_node_shared_ptr();
        const auto constant_input = std::dynamic_pointer_cast<ngraph::op::Constant>(input);
        if (constant_input != nullptr) {
            if (std::find(weight_ports.begin(), weight_ports.end(), i) != weight_ports.end()) {
                switch (node->get_element_type()) {
                    case ngraph::element::Type_t::boolean:
                        op_inputs.push_back(copy_constant_with_randomization<char>(constant_input));
                        break;
                    case ngraph::element::Type_t::bf16:
                        op_inputs.push_back(
                                copy_constant_with_randomization<ngraph::bfloat16>(constant_input));
                        break;
                    case ngraph::element::Type_t::f16:
                        op_inputs.push_back(
                                copy_constant_with_randomization<ngraph::float16>(constant_input));
                        break;
                    case ngraph::element::Type_t::f32:
                        op_inputs.push_back(copy_constant_with_randomization<float>(constant_input));
                        break;
                    case ngraph::element::Type_t::f64:
                        op_inputs.push_back(copy_constant_with_randomization<double>(constant_input));
                        break;
                    case ngraph::element::Type_t::i8:
                        op_inputs.push_back(copy_constant_with_randomization<int8_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::i16:
                        op_inputs.push_back(copy_constant_with_randomization<int16_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::i32:
                        op_inputs.push_back(copy_constant_with_randomization<int32_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::i64:
                        op_inputs.push_back(copy_constant_with_randomization<int64_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::u1:
                        op_inputs.push_back(copy_constant_with_randomization<char>(constant_input));
                        break;
                    case ngraph::element::Type_t::u8:
                        op_inputs.push_back(copy_constant_with_randomization<uint8_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::u16:
                        op_inputs.push_back(copy_constant_with_randomization<uint16_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::u32:
                        op_inputs.push_back(copy_constant_with_randomization<uint32_t>(constant_input));
                        break;
                    case ngraph::element::Type_t::u64:
                        op_inputs.push_back(copy_constant_with_randomization<uint64_t>(constant_input));
                        break;
                    default:
                        break;
                }
            } else {
                op_inputs.push_back(input->clone_with_new_inputs({}));
            }
        } else {
            op_inputs.push_back(std::make_shared<ngraph::op::Parameter>(input->get_element_type(),
                                                                        input->get_shape()));
        }
    }
    auto op_clone = node->clone_with_new_inputs(op_inputs);
    return op_clone;
}

template<>
const std::shared_ptr<ngraph::Node>
clone_with_new_inputs<ngraph::opset6::Convolution>(const std::shared_ptr<ngraph::opset6::Convolution> &node) {
    return clone_weightable_node(node, {1});
}

template<>
const std::shared_ptr<ngraph::Node>
clone_with_new_inputs<ngraph::opset6::GroupConvolution>(const std::shared_ptr<ngraph::opset6::GroupConvolution> &node) {
    return clone_weightable_node(node, {1});
}

template<>
const std::shared_ptr<ngraph::Node>
clone_with_new_inputs<ngraph::opset6::ConvolutionBackpropData>(
        const std::shared_ptr<ngraph::opset6::ConvolutionBackpropData> &node) {
    return clone_weightable_node(node, {1});
}
template<>
const std::shared_ptr<ngraph::Node>
clone_with_new_inputs<ngraph::opset6::GroupConvolutionBackpropData>(
        const std::shared_ptr<ngraph::opset6::GroupConvolutionBackpropData> &node) {
    return clone_weightable_node(node, {1});
}

}  // namespace SubgraphsDumper