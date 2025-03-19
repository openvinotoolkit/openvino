// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "factory.hpp"

#include "eltwise.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "reduce.hpp"

namespace ov::intel_cpu::tpp::op {
namespace {
struct CustomPowerStaticBuilder : public NodeFactory::TPPCustomBuilder {
    CustomPowerStaticBuilder() : NodeFactory::TPPCustomBuilder() {
        matcher = [](const std::shared_ptr<ov::Node>& n) {
            std::set<float> supported_power{-1, 2, 0.5};
            const auto& power_static = ov::as_type_ptr<const snippets::op::PowerStatic>(n);
            return power_static && supported_power.count(power_static->get_power());
        };
        builder = [](const std::shared_ptr<ov::Node>& n) {
            const auto& power_static = ov::as_type_ptr<snippets::op::PowerStatic>(n);
            OPENVINO_ASSERT(power_static, "Attempt to create TPP node from unsupported input in power_static_builder");
            const auto power = power_static->get_power();
            const auto& input = n->input_value(0);
            std::shared_ptr<ov::Node> tpp_node{nullptr};
            if (power == -1.f) {
                tpp_node = std::make_shared<Reciprocal>(input);
            } else if (power == 2.f) {
                tpp_node = std::make_shared<Square>(input);
            } else if (power == 0.5f) {
                tpp_node = std::make_shared<SquareRoot>(input);
            }
            OPENVINO_ASSERT(tpp_node, "Failed to create TPP in power_static_builder");
            return tpp_node;
        };
    }
};

}  // namespace
#define CREATE_UNARY_TPP_NODE(tpp_node_type)                                      \
    [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {      \
        return std::make_shared<tpp_node_type>(node->get_input_source_output(0)); \
    }

#define CREATE_BINARY_TPP_NODE(tpp_node_type)                                    \
    [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {     \
        return std::make_shared<tpp_node_type>(node->get_input_source_output(0), \
                                               node->get_input_source_output(1), \
                                               node->get_autob());               \
    }

#define CREATE_REDUCE_TPP_NODE(tpp_node_type)                                                           \
    [](const std::shared_ptr<ov::Node>& node) -> std::shared_ptr<ov::Node> {                            \
        const auto& reduce = ov::as_type_ptr<snippets::op::ReduceBase>(node);                           \
        OPENVINO_ASSERT(reduce, "Attempt to create TPP Reduce from invalid node");                      \
        return std::make_shared<tpp_node_type>(reduce->get_input_source_output(0), reduce->get_axis()); \
    }

std::unordered_map<ov::DiscreteTypeInfo, NodeFactory::tpp_builder> NodeFactory::m_direct_mapping{
    {ov::op::v1::Add::get_type_info_static(), CREATE_BINARY_TPP_NODE(Add)},
    {ov::op::v1::Subtract::get_type_info_static(), CREATE_BINARY_TPP_NODE(Subtract)},
    {ov::op::v1::Multiply::get_type_info_static(), CREATE_BINARY_TPP_NODE(Multiply)},
    {ov::op::v1::Divide::get_type_info_static(), CREATE_BINARY_TPP_NODE(Divide)},
    {ov::op::v0::Exp::get_type_info_static(), CREATE_UNARY_TPP_NODE(Exp)},
    {ov::op::v0::Relu::get_type_info_static(), CREATE_UNARY_TPP_NODE(Relu)},
    // Note that we don't support conversion from ngraph ops here, since they have a broader semantics (e.g. multiple
    // axis provided at a secont input)
    {ov::snippets::op::ReduceMax::get_type_info_static(), CREATE_REDUCE_TPP_NODE(ReduceMax)},
    {ov::snippets::op::ReduceSum::get_type_info_static(), CREATE_REDUCE_TPP_NODE(ReduceSum)},
};

std::vector<NodeFactory::TPPCustomBuilder> NodeFactory::m_custom_mapping{CustomPowerStaticBuilder()};

std::shared_ptr<ov::Node> NodeFactory::create(const std::shared_ptr<ov::Node>& n) {
    const auto& found = m_direct_mapping.find(n->get_type_info());
    std::shared_ptr<ov::Node> tpp_node{nullptr};
    if (found != m_direct_mapping.end()) {
        tpp_node = (found->second)(n);
    } else {
        for (const auto& custom_builder : m_custom_mapping) {
            if (custom_builder.matcher(n)) {
                tpp_node = custom_builder.builder(n);
                break;
            }
        }
    }
    return tpp_node;
}

bool NodeFactory::is_supported(const std::shared_ptr<ov::Node>& n) {
    auto matches = [=](const NodeFactory::TPPCustomBuilder& custom_builder) {
        return custom_builder.matcher(n);
    };
    // TPP currently supports only FP32 precisions (ticket: 130010)
    // Note: verify that TypeRelaxed property is maintained (mismatched input precisions)
    // after low precisions are enabled (ticket: 132328)
    const auto& ins = n->inputs();
    auto is_fp32_input = [](const ov::Input<ov::Node>& in) {
        return in.get_element_type() == element::f32;
    };
    const bool all_inputs_fp32 = std::all_of(ins.begin(), ins.end(), is_fp32_input);
    return (m_direct_mapping.count(n->get_type_info()) ||
            std::any_of(m_custom_mapping.begin(), m_custom_mapping.end(), matches)) &&
           all_inputs_fp32;
}

}  // namespace ov::intel_cpu::tpp::op
