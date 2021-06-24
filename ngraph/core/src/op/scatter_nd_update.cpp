// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/scatter_nd_update.hpp"
#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/scatter_nd_update.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::ScatterNDUpdate::type_info;

shared_ptr<Node> op::v3::ScatterNDUpdate::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_ScatterNDUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v3::ScatterNDUpdate>(new_args.at(op::util::ScatterNDBase::INPUTS),
                                                new_args.at(op::util::ScatterNDBase::INDICES),
                                                new_args.at(op::util::ScatterNDBase::UPDATES));
}

namespace scatter
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& arg2,
                  const HostTensorPtr& out)
    {
        using T = typename element_type_traits<ET>::value_type;
        Shape params_shape = arg0->get_shape();
        Shape indices_shape = arg1->get_shape();
        Shape updates_shape = arg1->get_shape();
        Shape out_shape(params_shape);
        out->set_shape(out_shape);

        if (arg1->get_element_type() == element::i64)
        {
            runtime::reference::scatterNdUpdate<T, int64_t>(arg0->get_data_ptr<ET>(),
                                                            arg1->get_data_ptr<int64_t>(),
                                                            arg2->get_data_ptr<ET>(),
                                                            out->get_data_ptr<ET>(),
                                                            arg0->get_shape(),
                                                            arg1->get_shape(),
                                                            arg2->get_shape());
        }
        else if (arg1->get_element_type() == element::i32)
        {
            runtime::reference::scatterNdUpdate<T, int32_t>(arg0->get_data_ptr<ET>(),
                                                            arg1->get_data_ptr<int32_t>(),
                                                            arg2->get_data_ptr<ET>(),
                                                            out->get_data_ptr<ET>(),
                                                            arg0->get_shape(),
                                                            arg1->get_shape(),
                                                            arg2->get_shape());
        }
        else
        {
            throw ngraph_error("Unexpected type");
        }

        return true;
    }

    bool evaluate_scatter(const HostTensorPtr& arg0,
                          const HostTensorPtr& arg1,
                          const HostTensorPtr& arg2,
                          const HostTensorPtr& out)
    {
        bool rc = true;

        switch (out->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_scatter, i32, arg0, arg1, arg2, out);
            NGRAPH_TYPE_CASE(evaluate_scatter, i64, arg0, arg1, arg2, out);
            NGRAPH_TYPE_CASE(evaluate_scatter, u32, arg0, arg1, arg2, out);
            NGRAPH_TYPE_CASE(evaluate_scatter, u64, arg0, arg1, arg2, out);
            NGRAPH_TYPE_CASE(evaluate_scatter, f16, arg0, arg1, arg2, out);
            NGRAPH_TYPE_CASE(evaluate_scatter, f32, arg0, arg1, arg2, out);
            NGRAPH_TYPE_CASE(evaluate_scatter, boolean, arg0, arg1, arg2, out);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace scatter

bool op::v3::ScatterNDUpdate::evaluate(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v3_ScatterNDUpdate_evaluate);
    NGRAPH_CHECK(!inputs.empty());
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 3));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));

    return scatter::evaluate_scatter(inputs[0], inputs[1], inputs[2], outputs[0]);
}

bool op::v3::ScatterNDUpdate::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v3_ScatterNDUpdate_has_evaluate);

    switch (get_output_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::boolean: break;
    default: return false;
    }
    switch (get_input_element_type(1))
    {
    case ngraph::element::i32:
    case ngraph::element::i64: break;
    default: return false;
    }
    return true;
}
