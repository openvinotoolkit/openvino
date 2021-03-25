// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_cloner.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "common_test_utils/data_utils.hpp"

namespace SubgraphsDumper {
namespace {
const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::Node> &node) {
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
    if (!data.empty()) {
        auto min_max = std::minmax_element(data.begin(), data.end());
        // Apply randomization only if constant stores several non-equal values
        if (ngraph::shape_size(const_node->get_shape()) != 1 &&
            *min_max.first - *min_max.second > std::numeric_limits<dType>::epsilon()) {
            CommonTestUtils::fill_vector<dType>(data, *min_max.first, *min_max.second);
        }
    }
    return std::make_shared<ngraph::op::Constant>(const_node->get_element_type(), const_node->get_shape(), data);
}


std::shared_ptr<ngraph::op::Constant>
copy_constant_with_randomization(const std::shared_ptr<ngraph::Node> &node,
                                 const std::shared_ptr<ngraph::op::Constant> &constant_input) {
    switch (node->get_element_type()) {
        case ngraph::element::Type_t::boolean:
            return copy_constant_with_randomization<char>(constant_input);
        case ngraph::element::Type_t::bf16:
            return copy_constant_with_randomization<ngraph::bfloat16>(constant_input);
        case ngraph::element::Type_t::f16:
            return copy_constant_with_randomization<ngraph::float16>(constant_input);
        case ngraph::element::Type_t::f32:
            return copy_constant_with_randomization<float>(constant_input);
        case ngraph::element::Type_t::f64:
            return copy_constant_with_randomization<double>(constant_input);
        case ngraph::element::Type_t::i8:
            return copy_constant_with_randomization<int8_t>(constant_input);
        case ngraph::element::Type_t::i16:
            return copy_constant_with_randomization<int16_t>(constant_input);
        case ngraph::element::Type_t::i32:
            return copy_constant_with_randomization<int32_t>(constant_input);
        case ngraph::element::Type_t::i64:
            return copy_constant_with_randomization<int64_t>(constant_input);
        case ngraph::element::Type_t::u1:
            return copy_constant_with_randomization<char>(constant_input);
        case ngraph::element::Type_t::u8:
            return copy_constant_with_randomization<uint8_t>(constant_input);
        case ngraph::element::Type_t::u16:
            return copy_constant_with_randomization<uint16_t>(constant_input);
        case ngraph::element::Type_t::u32:
            return copy_constant_with_randomization<uint32_t>(constant_input);
        case ngraph::element::Type_t::u64:
            return copy_constant_with_randomization<uint64_t>(constant_input);
        default:
            return {};
    }
}

std::shared_ptr<ngraph::Node> clone_weightable_node(const std::shared_ptr<ngraph::Node> &node,
                                                    const std::vector<size_t> &weight_ports) {
    ngraph::OutputVector op_inputs;
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto input = node->input(i).get_source_output().get_node_shared_ptr();
        const auto constant_input = std::dynamic_pointer_cast<ngraph::op::Constant>(input);
        if (!constant_input) {
            op_inputs.push_back(std::make_shared<ngraph::op::Parameter>(node->get_input_tensor(i).get_element_type(),
                                                                        node->get_input_tensor(i).get_shape()));
            continue;
        }
        if (std::find(weight_ports.begin(), weight_ports.end(), i) == weight_ports.end()) {
            op_inputs.push_back(input->clone_with_new_inputs({}));
            continue;
        }
        op_inputs.push_back(copy_constant_with_randomization(node, constant_input));
    }
    auto op_clone = node->clone_with_new_inputs(op_inputs);
    return op_clone;
}

// Clone nodes requiring weights randomization
const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::Convolution> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::GroupConvolution> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::ConvolutionBackpropData> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::GroupConvolutionBackpropData> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v0::MatMul> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::Add> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::Multiply> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::Subtract> &node) {
    return clone_weightable_node(node, {1});
}

const std::shared_ptr<ngraph::Node> clone(const std::shared_ptr<ngraph::op::v1::Power> &node) {
    return clone_weightable_node(node, {1});
}

template<typename opType>
const std::shared_ptr<ngraph::Node> clone_node(const std::shared_ptr<ngraph::Node> &node) {
    return clone(ngraph::as_type_ptr<opType>(node));
}
}  // namespace

#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, clone_node<NAMESPACE::NAME>},

const ClonersMap::cloners_map_type ClonersMap::cloners{
#include <ngraph/opsets/opset1_tbl.hpp>
#include <ngraph/opsets/opset2_tbl.hpp>
#include <ngraph/opsets/opset3_tbl.hpp>
#include <ngraph/opsets/opset4_tbl.hpp>
#include <ngraph/opsets/opset5_tbl.hpp>
#include <ngraph/opsets/opset6_tbl.hpp>
};
#undef NGRAPH_OP

}  // namespace SubgraphsDumper
