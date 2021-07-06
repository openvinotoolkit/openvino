// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/not_equal.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/not_equal.hpp"

using namespace std;
using namespace ngraph;

namespace not_equalop
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::not_equal(arg0->get_data_ptr<ET>(),
                                      arg1->get_data_ptr<ET>(),
                                      out->get_data_ptr<element::Type_t::boolean>(),
                                      arg0->get_shape(),
                                      arg1->get_shape(),
                                      broadcast_spec);
        return true;
    }

    bool evaluate_not_equal(const HostTensorPtr& arg0,
                            const HostTensorPtr& arg1,
                            const HostTensorPtr& out,
                            const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1, element::boolean);
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_not_equal, boolean, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_not_equal, i32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_not_equal, i64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_not_equal, u32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_not_equal, u64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_not_equal, f16, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_not_equal, f32, arg0, arg1, out, broadcast_spec);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace not_equalop

// ----------------------------------- v1 --------------------------------------

NGRAPH_RTTI_DEFINITION(op::v1::NotEqual, "NotEqual", 1);

op::v1::NotEqual::NotEqual(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseComparison(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::NotEqual::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_NotEqual_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::NotEqual>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::NotEqual::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_NotEqual_evaluate);
    return not_equalop::evaluate_not_equal(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::NotEqual::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_NotEqual_has_evaluate);
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

bool op::v1::NotEqual::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_NotEqual_visit_attributes);
    return true;
}
