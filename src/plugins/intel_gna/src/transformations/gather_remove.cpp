// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/gather_remove.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include "ngraph/validation_util.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

using namespace ov::intel_gna::pass;
using namespace GNAPluginNS;

using Node = std::shared_ptr<ngraph::Node>;
using Function = std::shared_ptr<ngraph::Function>;

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

    Node node_parent = node->get_input_node_shared_ptr(0);
    if (!std::equal(input_node_shape.begin(), input_node_shape.end(), output_node_shape.begin())) {
        auto reshape_const_node = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
                                                                             ngraph::Shape{output_node_shape.size()},
                                                                             output_node_shape);
        node_parent = std::make_shared<ngraph::opset8::Reshape>(node_parent, reshape_const_node, false);
    }

    node->output(0).replace(node_parent->output(0));
}

/*
  Support only one data node as 0 input
 */
Function CopySingleInputNodeFromFunction(Node node) {
    const ngraph::Shape & input_shape = node->get_input_shape(0);
    const ngraph::element::Type& input_elem_type = node->get_input_element_type(0);

    auto input_params = std::make_shared<ngraph::opset8::Parameter>(input_elem_type, input_shape);
    auto input_nodes = node->input_values();
    input_nodes[0] = input_params;
    auto node_copy = node->clone_with_new_inputs(input_nodes);
    auto result = std::make_shared<ngraph::opset8::Result>(node_copy);

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                           ngraph::ParameterVector{input_params});
}

} // namespace

namespace GatherParamsRemoveNS {
    void DoTransformation(Node param_node, Node gather_node, SubgraphCPUMap * subgraph_cpu_map);

    void DoTransformation(Node param_node, Node gather_node, SubgraphCPUMap * subgraph_cpu_map /* may be nullptr */) {
        if (subgraph_cpu_map)
            subgraph_cpu_map->emplace(param_node->get_friendly_name(), CopySingleInputNodeFromFunction(gather_node));

        RemoveSingleInputNodeFromFunction(gather_node);
    }
 } // namespace GatherParamsRemoveNS

namespace GatherResultRemoveNS {
    void DoTransformation(Node gather_node, Node result_node, SubgraphCPUMap * subgraph_cpu_map);

    void DoTransformation(Node gather_node, Node result_node, SubgraphCPUMap * subgraph_cpu_map /* may be nullptr */) {
        if (subgraph_cpu_map) {
            const std::string & parent_name = gather_node->get_input_node_shared_ptr(0)->get_friendly_name();
            subgraph_cpu_map->emplace(parent_name, CopySingleInputNodeFromFunction(gather_node));
        }

        RemoveSingleInputNodeFromFunction(gather_node);
    }
} // namespace GatherResultRemoveNS

// ----------------------------------------------------------------------------

GatherResultRemove::GatherResultRemove(SubgraphCPUMap * subgraph_cpu_map)
    : m_subgraph_cpu_map(subgraph_cpu_map) {

    MATCHER_SCOPE(GatherResultRemove);

    auto gather = ngraph::pattern::wrap_type<ngraph::opset8::Gather>({ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input()});
    auto result = ngraph::pattern::wrap_type<ngraph::opset8::Result>({gather});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto result_node = pattern_map.at(result).get_node_shared_ptr();
        const auto gather_node = pattern_map.at(gather).get_node_shared_ptr();

        GatherResultRemoveNS::DoTransformation(gather_node, result_node, m_subgraph_cpu_map);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

GatherParamsRemove::GatherParamsRemove(SubgraphCPUMap * subgraph_cpu_map)
    : m_subgraph_cpu_map(subgraph_cpu_map) {

    MATCHER_SCOPE(GatherParamsRemove);

    auto param = ngraph::pattern::wrap_type<ngraph::opset8::Parameter>();
    auto gather = ngraph::pattern::wrap_type<ngraph::opset8::Gather>({param,
                                                                      ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input()});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto param_node = pattern_map.at(param).get_node_shared_ptr();
        const auto gather_node = pattern_map.at(gather).get_node_shared_ptr();

        GatherParamsRemoveNS::DoTransformation(param_node, gather_node, m_subgraph_cpu_map);
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
