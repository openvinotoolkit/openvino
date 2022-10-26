// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <istream>
#include "op_cloner.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "common_test_utils/data_utils.hpp"

namespace SubgraphsDumper {
namespace {


template<typename dType>
void get_port_range(const std::shared_ptr<ov::op::v0::Constant> &const_node, LayerTestsUtils::PortInfo &port_info) {
    std::vector<dType> data = const_node->cast_vector<dType>();
    if (!data.empty()) {
        auto min_max = std::minmax_element(data.begin(), data.end());
        port_info.min = *min_max.first;
        port_info.max = *min_max.second;
    }
}


void get_port_range(const std::shared_ptr<ov::op::v0::Constant> &constant_input, LayerTestsUtils::PortInfo &port_info) {
    switch (constant_input->get_element_type()) {
        case ov::element::Type_t::boolean:
            get_port_range<char>(constant_input, port_info);
            break;
        case ov::element::Type_t::bf16:
            get_port_range<ov::bfloat16>(constant_input, port_info);
            break;
        case ov::element::Type_t::f16:
            get_port_range<ov::float16>(constant_input, port_info);
            break;
        case ov::element::Type_t::f32:
            get_port_range<float>(constant_input, port_info);
            break;
        case ov::element::Type_t::f64:
            get_port_range<double>(constant_input, port_info);
            break;
        case ov::element::Type_t::i8:
            get_port_range<int8_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::i16:
            get_port_range<int16_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::i32:
            get_port_range<int32_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::i64:
            get_port_range<int64_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::u1:
            get_port_range<char>(constant_input, port_info);
            break;
        case ov::element::Type_t::u8:
            get_port_range<uint8_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::u16:
            get_port_range<uint16_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::u32:
            get_port_range<uint32_t>(constant_input, port_info);
            break;
        case ov::element::Type_t::u64:
            get_port_range<uint64_t>(constant_input, port_info);
            break;
        default:
            break;
    }
}

std::shared_ptr<ov::Node> clone(const std::shared_ptr<ov::Node> &node, LayerTestsUtils::OPInfo &meta) {
    ov::OutputVector op_inputs;
    bool has_parameters = false;
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto input = node->input(i).get_source_output();
        auto port_info = LayerTestsUtils::PortInfo();
        const auto constant = ov::get_constant_from_source(input);
        if (constant) {
            get_port_range(constant, port_info);
            float weights_size =
                    static_cast<float>(ov::shape_size(constant->get_shape()) *
                                       constant->get_element_type().size()) / (1024 * 1024);
            if (weights_size > ClonersMap::constant_size_threshold_mb) {
                std::cout << "Constant with size " << weights_size << " detected on port " << i << " of OP " << node
                          << std::endl
                          << "The constant will be replaced with parameter and initial data ranges meta info"
                          << std::endl;
                auto param = std::make_shared<ov::op::v0::Parameter>(constant->get_element_type(),
                                                                                           constant->get_shape());
                op_inputs.push_back(param);

                has_parameters = true;

            } else {
                const auto clone = std::make_shared<ov::op::v0::Constant>(constant->get_element_type(),
                                                                              constant->get_shape(),
                                                                              constant->get_data_ptr());
                op_inputs.push_back(clone);
            }
        } else {
            has_parameters = true;
            auto param = std::make_shared<ov::op::v0::Parameter>(input.get_element_type(),
                                                                 input.get_partial_shape());
            op_inputs.push_back(param);
        }
        meta.ports_info[i] = port_info;
    }
    if (!has_parameters) {
        return nullptr;
    }
    return node->clone_with_new_inputs(op_inputs);
}

std::shared_ptr<ov::Node> clone_weightable_node(const std::shared_ptr<ov::Node> &node,
                                                const std::vector<size_t> &weight_ports,
                                                LayerTestsUtils::OPInfo &meta) {
    ov::OutputVector op_inputs;
    bool has_parameters = false;
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        const auto input = node->input(i).get_source_output();
        const auto constant_input = ov::get_constant_from_source(input);
        auto port_info = LayerTestsUtils::PortInfo();
        // Input is Parameter or dynamic data pass
        if (!constant_input) {
            has_parameters = true;
            auto param = std::make_shared<ov::op::v0::Parameter>(input.get_element_type(),
                                                                 input.get_partial_shape());
            op_inputs.push_back(param);
            meta.ports_info[i] = port_info;
            continue;
        }
        get_port_range(constant_input, port_info);
        // Input is Constant but not in the target weight ports
        if (std::find(weight_ports.begin(), weight_ports.end(), i) == weight_ports.end()) {
            float weights_size =
                    static_cast<float>(ov::shape_size(constant_input->get_shape()) *
                                       constant_input->get_element_type().size()) / (1024 * 1024);
            if (weights_size > ClonersMap::constant_size_threshold_mb) {
                std::cout << "Constant with size " << weights_size << " detected on port " << i << " of OP " << node
                          << std::endl
                          << "The constant will be replaced with parameter and initial data ranges meta info"
                          << std::endl;
                auto param = std::make_shared<ov::op::v0::Parameter>(constant_input->get_element_type(),
                                                                                   constant_input->get_shape());
                op_inputs.push_back(param);

                has_parameters = true;
            } else {
                const auto clone = std::make_shared<ov::op::v0::Constant>(constant_input->get_element_type(),
                                                                              constant_input->get_shape(),
                                                                              constant_input->get_data_ptr());
                op_inputs.push_back(clone);
            }
            meta.ports_info[i] = port_info;
            continue;
        }
        // Input is constant and in the target weights ports
        auto param = std::make_shared<ov::op::v0::Parameter>(constant_input->get_element_type(),
                                                                          constant_input->get_shape());
        port_info.convert_to_const = true;
        meta.ports_info[i] = port_info;
        op_inputs.push_back(param);
    }
    if (!has_parameters) {
        return nullptr;
    }
    auto op_clone = node->clone_with_new_inputs(op_inputs);
    return op_clone;
}

// Clone nodes requiring weights randomization
std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v1::Convolution> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {1}, meta);
}

std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v1::GroupConvolution> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {1}, meta);
}

std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v1::ConvolutionBackpropData> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {1}, meta);
}

std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {1}, meta);
}

std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v0::MatMul> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {0, 1}, meta);
}

std::shared_ptr<ov::Node> clone(const std::shared_ptr<ov::op::v1::Add> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {0, 1}, meta);
}

std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v1::Multiply> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {0, 1}, meta);
}

std::shared_ptr<ov::Node>
clone(const std::shared_ptr<ov::op::v1::Subtract> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {0, 1}, meta);
}

std::shared_ptr<ov::Node> clone(const std::shared_ptr<ov::op::v1::Power> &node, LayerTestsUtils::OPInfo &meta) {
    return clone_weightable_node(node, {0, 1}, meta);
}

template<typename opType>
std::shared_ptr<ov::Node> clone_node(const std::shared_ptr<ov::Node> &node, LayerTestsUtils::OPInfo &meta) {
    return clone(ov::as_type_ptr<opType>(node), meta);
}
}  // namespace

#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), clone_node<NAMESPACE::NAME>},

const ClonersMap::cloners_map_type ClonersMap::cloners{
#include <ngraph/opsets/opset1_tbl.hpp>
#include <ngraph/opsets/opset2_tbl.hpp>
#include <ngraph/opsets/opset3_tbl.hpp>
#include <ngraph/opsets/opset4_tbl.hpp>
#include <ngraph/opsets/opset5_tbl.hpp>
#include <ngraph/opsets/opset6_tbl.hpp>
#include <ngraph/opsets/opset7_tbl.hpp>
#include <ngraph/opsets/opset8_tbl.hpp>
#include <ngraph/opsets/opset9_tbl.hpp>
};
#undef NGRAPH_OP

float ClonersMap::constant_size_threshold_mb = 0.5;
}  // namespace SubgraphsDumper
