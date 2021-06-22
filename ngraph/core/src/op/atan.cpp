// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/op/atan.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/atan.hpp"

#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Atan::type_info;

op::Atan::Atan(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Atan::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Atan_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Atan>(new_args.at(0));
}

namespace atanop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::atan<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_atan(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_atan, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_atan, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_atan, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_atan, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_atan, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_atan, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_atan, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace atanop

bool op::Atan::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Atan_evaluate);
    return atanop::evaluate_atan(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Atan::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Atan_has_evaluate);
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
