// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "itt.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------------ v0 -------------------------------------

namespace maximumop
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  const op::AutoBroadcastSpec& broadcast_spec)
    {
        runtime::reference::maximum(arg0->get_data_ptr<ET>(),
                                    arg1->get_data_ptr<ET>(),
                                    out->get_data_ptr<ET>(),
                                    arg0->get_shape(),
                                    arg1->get_shape(),
                                    broadcast_spec);
        return true;
    }

    bool evaluate_maximum(const HostTensorPtr& arg0,
                          const HostTensorPtr& arg1,
                          const HostTensorPtr& out,
                          const op::AutoBroadcastSpec& broadcast_spec)
    {
        bool rc = true;
        out->set_broadcast(broadcast_spec, arg0, arg1);
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_maximum, i32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_maximum, i64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_maximum, u32, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_maximum, u64, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_maximum, f16, arg0, arg1, out, broadcast_spec);
            NGRAPH_TYPE_CASE(evaluate_maximum, f32, arg0, arg1, out, broadcast_spec);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace maximumop

// ------------------------------------ v1 -------------------------------------

constexpr NodeTypeInfo op::v1::Maximum::type_info;

op::v1::Maximum::Maximum(const Output<Node>& arg0,
                         const Output<Node>& arg1,
                         const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Maximum::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Maximum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Maximum>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::Maximum::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Maximum_evaluate);
    return maximumop::evaluate_maximum(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::Maximum::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Maximum_has_evaluate);
    switch (get_input_element_type(0))
    {
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
