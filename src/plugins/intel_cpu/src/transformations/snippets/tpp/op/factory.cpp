// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "factory.hpp"
#include "eltwise.hpp"
#include "reduce.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {
namespace {
struct CustomPowerStaticBuilder : public TPPNodeFactory::TPPCustomBuilder {
    CustomPowerStaticBuilder() : TPPNodeFactory::TPPCustomBuilder() {
        matcher = [](const std::shared_ptr<ov::Node>& n) {
            std::set<float> supported_power{-1, 2, 0.5};
            const auto& power_static = ov::as_type_ptr<const snippets::op::PowerStatic>(n);
            return power_static && supported_power.count(power_static->get_power());
        };
        builder = [](const std::shared_ptr<ov::Node>& n) {
            const auto& power_static = ov::as_type_ptr<snippets::op::PowerStatic>(n);
            OPENVINO_ASSERT(power_static, "Attempt to create TPP node from unsupported input in power_static_builder");
            const auto power = power_static->get_power();
            const auto& input = n->get_input_source_output(0);
            std::shared_ptr<ov::Node> tpp_node{nullptr};
            if (power == -1.f)
                tpp_node = std::make_shared<Reciprocal>(input);
            else if (power == 2.f)
                tpp_node = std::make_shared<Square>(input);
            else if (power == 0.5f)
                tpp_node = std::make_shared<SquareRoot>(input);
            OPENVINO_ASSERT(tpp_node, "Failed to create TPP in power_static_builder");
            return tpp_node;
        };
    }
};

} // namespace
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

#define CREATE_REDUCE_TPP_NODE(tpp_node_type) \
    [](const std::shared_ptr<ov::Node>& ngraph_node) -> std::shared_ptr<ov::Node> { \
        const auto& reduce = ov::as_type_ptr<snippets::op::ReduceBase>(ngraph_node); \
        OPENVINO_ASSERT(reduce, "Attempt to create TPP Reduce from invalid node"); \
        return std::make_shared<tpp_node_type>(reduce->get_input_source_output(0), reduce->get_axis()); \
    }

    std::unordered_map<ov::DiscreteTypeInfo, TPPNodeFactory::tpp_builder> TPPNodeFactory::m_direct_mappig {
        {ov::op::v1::Add::get_type_info_static(), CREATE_BINARY_TPP_NODE(Add)},
        {ov::op::v1::Subtract::get_type_info_static(), CREATE_BINARY_TPP_NODE(Subtract)},
        {ov::op::v1::Multiply::get_type_info_static(), CREATE_BINARY_TPP_NODE(Multiply)},
        {ov::op::v1::Divide::get_type_info_static(), CREATE_BINARY_TPP_NODE(Divide)},
        {ov::op::v0::Exp::get_type_info_static(), CREATE_UNARY_TPP_NODE(Exp)},
        {ov::op::v0::Relu::get_type_info_static(), CREATE_UNARY_TPP_NODE(Relu)},
        // Note that we don't support conversion from ngraph ops here, since they have a broader semantics (e.g. multiple axis provided at a secont input)
        {ov::snippets::op::ReduceMax::get_type_info_static(), CREATE_REDUCE_TPP_NODE(ReduceMax)},
        {ov::snippets::op::ReduceSum::get_type_info_static(), CREATE_REDUCE_TPP_NODE(ReduceSum)},
    };


    std::vector<TPPNodeFactory::TPPCustomBuilder> TPPNodeFactory::m_custom_mapping{CustomPowerStaticBuilder()};
    std::shared_ptr<ov::Node> TPPNodeFactory::create(const std::shared_ptr<ov::Node>& n) {
        const auto& found = m_direct_mappig.find(n->get_type_info());
        std::shared_ptr<ov::Node> tpp_node{nullptr};
        if (found != m_direct_mappig.end()) {
            tpp_node = (found->second)(n);
        } else {
            for (const auto& custom_builder : m_custom_mapping) {
                if (custom_builder.matcher(n)) {
                    tpp_node = custom_builder.builder(n);
                    break;
                }
            }
        }
        if (tpp_node)
            tpp_node->set_friendly_name(n->get_friendly_name());
        return tpp_node;
    }

    bool TPPNodeFactory::is_supported(const std::shared_ptr<ov::Node>& n) {
        auto matches = [=](const TPPNodeFactory::TPPCustomBuilder& custom_builder) {
            return custom_builder.matcher(n);
        };
        return m_direct_mappig.count(n->get_type_info()) ||
               std::any_of(m_custom_mapping.begin(), m_custom_mapping.end(), matches);
    }

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
