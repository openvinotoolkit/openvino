// Copyright (C) 2022 Intel CorporationNodePtr
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/gather_remove.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <legacy/ngraph_ops/gather_ie.hpp>
#include "ngraph/validation_util.hpp"
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/or.hpp>

using namespace ov;
using namespace ov::intel_gna::pass;

using NodePtr = std::shared_ptr<ngraph::Node>;
using Function = std::shared_ptr<ngraph::Function>;

namespace {

void SwapOutputNames(Output<Node> output1, Output<Node> output2) {
    const auto node2_output_names = output2.get_names();
    output2.set_names(output1.get_names());
    output1.set_names(node2_output_names);
}

template <typename NodePtr1, typename NodePtr2>
void SwapFriendlyNames(NodePtr1 node1, NodePtr2 node2) {
    const std::string node2_name = node2->get_friendly_name();
    node2->set_friendly_name(node1->get_friendly_name());
    node1->set_friendly_name(node2_name);
}

template <typename NodePtr1, typename NodePtr2>
void SwapNames(NodePtr1 node1, NodePtr2 node2) {
    SwapFriendlyNames(node1, node2);
    SwapOutputNames(node1->output(0), node2->output(0));
}

class GatherResultRemove : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherResultRemove(ov::intel_gna::SubgraphCPUMap * subgraph_cpu_map = nullptr);
private:
    ov::intel_gna::SubgraphCPUMap * m_subgraph_cpu_map;
};

class GatherParamsRemove : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GatherParamsRemove(ov::intel_gna::SubgraphCPUMap * subgraph_cpu_map = nullptr);
private:
    ov::intel_gna::SubgraphCPUMap * m_subgraph_cpu_map;
};

} // namespace

NGRAPH_RTTI_DEFINITION(GatherResultRemove, "GatherResultRemove", 0);
NGRAPH_RTTI_DEFINITION(GatherParamsRemove, "GatherParamsRemove", 0);
NGRAPH_RTTI_DEFINITION(GatherRemove, "GatherRemove", 0);

namespace {

/*
  works only if we have one date input and one output
 */
void RemoveSingleInputNodeFromFunction(std::shared_ptr<ngraph::Node> node) {
    const ngraph::Shape input_node_shape = node->get_input_shape(0);
    const ngraph::Shape output_node_shape = node->get_output_shape(0);

    NodePtr node_parent = node->get_input_node_shared_ptr(0);
    if (!std::equal(input_node_shape.begin(), input_node_shape.end(), output_node_shape.begin())) {
        auto reshape_const_node = std::make_shared<ngraph::opset9::Constant>(ngraph::element::Type_t::i64,
                                                                             ngraph::Shape{output_node_shape.size()},
                                                                             output_node_shape);
        node_parent = std::make_shared<ngraph::opset9::Reshape>(node_parent, reshape_const_node, false);
    }

    node->output(0).replace(node_parent->output(0));
}

/*
  Support only one data node as 0 input
 */
Function CopySingleInputNodeFromFunction(NodePtr node) {
    const ngraph::Shape & input_shape = node->get_input_shape(0);
    const ngraph::element::Type& input_elem_type = ngraph::element::Type_t::f32;

    auto input_params = std::make_shared<ngraph::opset9::Parameter>(input_elem_type, input_shape);
    auto input_nodes = node->input_values();
    input_nodes[0] = input_params;
    auto node_copy = node->clone_with_new_inputs(input_nodes);
    auto result = std::make_shared<ngraph::opset9::Result>(node_copy);

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
}

} // namespace

// ----------------------------------------------------------------------------

GatherResultRemove::GatherResultRemove(SubgraphCPUMap * subgraph_cpu_map)
    : m_subgraph_cpu_map(subgraph_cpu_map) {

    MATCHER_SCOPE(GatherResultRemove);

    auto gather = ngraph::pattern::wrap_type<ngraph::opset9::Gather, ngraph::op::GatherIE>({ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input()}); // FIXME: add consumers(1) constraint
    auto result = ngraph::pattern::wrap_type<ngraph::opset9::Result>({gather});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto result_node = pattern_map.at(result).get_node_shared_ptr();
        const auto gather_node = pattern_map.at(gather).get_node_shared_ptr();

        NodePtr parent_node = gather_node->get_input_node_shared_ptr(0);

        if (m_subgraph_cpu_map) {
            const std::string & gather_name = gather_node->get_friendly_name();
            m_subgraph_cpu_map->emplace(gather_name, CopySingleInputNodeFromFunction(gather_node));
        }
        RemoveSingleInputNodeFromFunction(gather_node);

        SwapNames(gather_node, parent_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

GatherParamsRemove::GatherParamsRemove(SubgraphCPUMap * subgraph_cpu_map)
    : m_subgraph_cpu_map(subgraph_cpu_map) {

    MATCHER_SCOPE(GatherParamsRemove);

    auto param = ngraph::pattern::wrap_type<ngraph::opset9::Parameter>();
    auto gather = ngraph::pattern::wrap_type<ngraph::opset9::Gather>({param,
                                                                      ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input()}); // FIXME: add consumers(1) constraint
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto param_node = pattern_map.at(param).get_node_shared_ptr();
        const auto gather_node = pattern_map.at(gather).get_node_shared_ptr();

        Node * child_node = gather_node->output(0).get_target_inputs().begin()->get_node();

        if (m_subgraph_cpu_map)
            m_subgraph_cpu_map->emplace(param_node->get_friendly_name(), CopySingleInputNodeFromFunction(gather_node));
        RemoveSingleInputNodeFromFunction(gather_node);

        SwapNames(child_node, gather_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, matcher_name);
    this->register_matcher(m, callback);
}

bool GatherRemove::run_on_model(const std::shared_ptr<ngraph::Function> & function) {
    RUN_ON_FUNCTION_SCOPE(GatherRemove);

    ngraph::pass::Manager manager(get_pass_config());
    manager.register_pass<GatherResultRemove>(m_subgraph_cpu_map);
    manager.register_pass<GatherParamsRemove>(m_subgraph_cpu_map);
    manager.run_passes(function);

    return false; // FIXME: should we return true here?
}
