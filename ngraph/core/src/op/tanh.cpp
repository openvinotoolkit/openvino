// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/tanh.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/tanh.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Tanh::type_info;

op::Tanh::Tanh(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Tanh::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Tanh_visit_attributes);
    return true;
}

shared_ptr<Node> op::Tanh::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Tanh_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Tanh>(new_args.at(0));
}

namespace tanhop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::tanh<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_tanh(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_tanh, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_tanh, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_tanh, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_tanh, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_tanh, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_tanh, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace tanhop

bool op::Tanh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Tanh_evaluate);
    return tanhop::evaluate_tanh(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Tanh::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Tanh_has_evaluate);
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
