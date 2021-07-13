// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/my_node.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/my_node.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::MyNode::type_info;

op::MyNode::MyNode(const Output<Node>& arg)
        : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::MyNode::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_MyNode_visit_attributes);
    return true;
}

shared_ptr<Node> op::MyNode::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_MyNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<MyNode>(new_args.at(0));
}

namespace myop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::my_node<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_my_node(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_my_node, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_my_node, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_my_node, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_my_node, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_my_node, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_my_node, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_my_node, f32, arg0, out, count);
            default: rc = false; break;
        }
        return rc;
    }
} // namespace myop

bool op::MyNode::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_MyNode_evaluate);
    return myop::evaluate_my_node(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::MyNode::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_MyNode_has_evaluate);
    switch (get_input_element_type(0))
    {
        case ngraph::element::boolean:
        case ngraph::element::i32:
        case ngraph::element::i64:
        case ngraph::element::u32:
        case ngraph::element::u64:
        case ngraph::element::f16:
        case ngraph::element::f32: return true;
        default: break;
    }
    return false;
}

