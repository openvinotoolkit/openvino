// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/op/asin.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/asin.hpp"

#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Asin::type_info;

op::Asin::Asin(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Asin::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Asin_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Asin>(new_args.at(0));
}

namespace asinop
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::asin<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_asin(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_asin, boolean, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_asin, i32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_asin, i64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_asin, u32, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_asin, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_asin, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_asin, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace asinop

bool op::Asin::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Asin_evaluate);
    return asinop::evaluate_asin(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool op::Asin::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Asin_has_evaluate);
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
