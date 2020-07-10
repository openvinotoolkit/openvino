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

#include "ngraph/op/util/fused_op.hpp"

#include "ngraph/graph_util.hpp"

using namespace ngraph;

op::util::FusedOp::FusedOp()
    : Op()
{
}

op::util::FusedOp::FusedOp(const OutputVector& args)
    : Op(args)
{
}

void op::util::FusedOp::validate_and_infer_types()
{
    pre_validate_and_infer_types();

    if (!can_decompose_with_partial_shapes() && is_dynamic())
    {
        return;
    }

    auto subgraph_outputs = decompose_op();
    //auto subgraph = extract_subgraph(as_output_vector(subgraph_outputs), input_values());
    auto subgraph = extract_subgraph(subgraph_outputs, get_arguments());
    validate_nodes_and_infer_types(subgraph);

    size_t i = 0;
    for (const auto& output_node : subgraph_outputs)
    {
        for (size_t j = 0; j < output_node->get_output_size(); j++, i++)
        {
            if (i >= get_output_size())
            {
                set_output_size(i + 1);
            }
            set_output_type(
                i, output_node->get_output_element_type(j), output_node->get_output_shape(j));
        }
    }

    post_validate_and_infer_types();
}

void op::util::FusedOp::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                          const OutputVector& /* deltas */)
{
    // TODO
    throw ngraph_error("Autodiff on fused ops not supported yet");
}
