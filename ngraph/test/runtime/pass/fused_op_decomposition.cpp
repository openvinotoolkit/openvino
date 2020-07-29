//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "fused_op_decomposition.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/provenance.hpp"

using namespace std;
using namespace ngraph;

pass::FusedOpDecomposition::FusedOpDecomposition(op_query_t callback)
    : m_has_direct_support{callback}
{
}

bool pass::FusedOpDecomposition::run_on_node(shared_ptr<Node> node)
{
    bool modified = false;

    if (op::supports_decompose(node))
    {
        if (m_has_direct_support && m_has_direct_support(*node))
        {
            // Op supported by backend. Do not decompose
            return modified;
        }

        OutputVector output_vector = node->decompose_op();
        NodeVector subgraph_outputs = as_node_vector(output_vector);

        if (ngraph::get_provenance_enabled())
        {
            // Capture the input values as an edge for provenance
            auto base_input_values = node->input_values();
            auto provenance_tags = node->get_provenance_tags();
            const std::string tag = "<Decomposed from " + std::string(node->get_type_name()) + ">";
            provenance_tags.insert(tag);

            // Transfer the new provenance tags to the newly created ops
            for (auto output_node : subgraph_outputs)
            {
                output_node->add_provenance_tags_above(base_input_values, provenance_tags);
            }
        }

        // Run recursively until no more fused ops
        auto subgraph = extract_subgraph(subgraph_outputs, as_node_vector(node->input_values()));
        for (auto subgraph_node : subgraph)
        {
            run_on_node(subgraph_node);
        }

        size_t i = 0;
        for (auto output_node : subgraph_outputs)
        {
            for (size_t j = 0; j < output_node->outputs().size(); j++, i++)
            {
                std::set<Input<Node>> fop_users = node->outputs().at(i).get_target_inputs();
                for (auto fop_user : fop_users)
                {
                    if (auto goe = as_type<op::GetOutputElement>(fop_user.get_node()))
                    {
                        Output<Node> goe_output = goe->get_as_output();
                        if (goe_output.get_index() == i &&
                            !goe->output(0).get_target_inputs().empty())
                        {
                            // Replace GOE users
                            std::set<Input<Node>> goe_users =
                                goe->outputs().at(0).get_target_inputs();
                            for (auto goe_user : goe_users)
                            {
                                goe_user.replace_source_output(output_node->output(j));
                            }
                        }
                    }
                    else
                    {
                        fop_user.replace_source_output(output_node->output(j));
                    }
                }
            }
        }
        if (i != node->get_output_size())
        {
            throw ngraph_error("While replacing " + node->get_name() +
                               ", mismatch between op output count and outputs of the decomposed "
                               "subgraph. Expected: " +
                               to_string(node->get_output_size()) + " Got: " + to_string(i));
        }
        modified = true;
    }

    return modified;
}
