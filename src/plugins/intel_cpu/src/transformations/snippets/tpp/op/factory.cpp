// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "factory.hpp"
#include "eltwise.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
// TODO: it might be more convenient to use template with a non-type parameter that describes the number of arguments and to provide
// partial specialization of the template for 1 and args. Note however that functional templates do not support partial specialization,
// do we will need to create a dedicated class for that. Discuss if the templated solution is better on review.
#define CREATE_UNARY_TPP_NODE(tpp_node_type) \
    [](const std::shared_ptr<ov::Node>& ngraph_node) -> std::shared_ptr<ov::Node> { \
        return std::make_shared<tpp_node_type>(ngraph_node->get_input_source_output(0)); \
    }

#define CREATE_BINARY_TPP_NODE(tpp_node_type) \
    [](const std::shared_ptr<ov::Node>& ngraph_node) -> std::shared_ptr<ov::Node> { \
        return std::make_shared<tpp_node_type>(ngraph_node->get_input_source_output(0), ngraph_node->get_input_source_output(1), ngraph_node->get_autob()); \
    }

    std::unordered_map<ov::DiscreteTypeInfo, TPPNodeFactory::tpp_creator> TPPNodeFactory::m_supported {
        {ov::op::v1::Add::get_type_info_static(), CREATE_BINARY_TPP_NODE(Add)},
        {ov::op::v1::Subtract::get_type_info_static(), CREATE_BINARY_TPP_NODE(Subtract)},
        {ov::op::v1::Multiply::get_type_info_static(), CREATE_BINARY_TPP_NODE(Multiply)},
        {ov::op::v1::Divide::get_type_info_static(), CREATE_BINARY_TPP_NODE(Divide)},
        {ov::op::v0::Exp::get_type_info_static(), CREATE_UNARY_TPP_NODE(Exp)},
        {ov::op::v0::Relu::get_type_info_static(), CREATE_UNARY_TPP_NODE(Relu)},
    };

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
