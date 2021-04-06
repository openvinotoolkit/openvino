// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/framework_node.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::FrameworkNode, "FrameworkNode", 0);

op::util::FrameworkNode::FrameworkNode(const OutputVector& inputs)
    : Op(inputs)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::util::FrameworkNode::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(FrameworkNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto node = std::make_shared<op::util::FrameworkNode>(new_args);
    for (size_t i = 0; i < get_output_size(); ++i)
    {
        node->set_output_type(i, get_output_element_type(i), get_output_partial_shape(i));
    }
    return node;
}

void op::util::FrameworkNode::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_FrameworkNode_validate_and_infer_types);
    // Save initial inputs descriptors
    for (uint64_t i = 0; i < get_input_size(); i++)
    {
        // TODO: store constant values
        const auto& new_input_desc =
            std::make_tuple(get_input_partial_shape(i), get_input_element_type(i));

        if (m_inputs_desc.empty())
        {
            m_inputs_desc.push_back(new_input_desc);
        }
        else
        {
            NODE_VALIDATION_CHECK(this,
                                  m_inputs_desc[i] == new_input_desc,
                                  "Input descriptor for ",
                                  get_friendly_name(),
                                  "has been changed.");
        }
    }
}

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<op::util::FrameworkNodeAttrs>::type_info;

    AttributeAdapter<op::util::FrameworkNodeAttrs>::AttributeAdapter(
        op::util::FrameworkNodeAttrs& value)
        : DirectValueAccessor<op::util::FrameworkNodeAttrs>(value)
    {
    }
}
