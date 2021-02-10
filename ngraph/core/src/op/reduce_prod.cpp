//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/reduce_prod.hpp"
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::ReduceProd::type_info;

op::v1::ReduceProd::ReduceProd(const Output<Node>& arg,
                               const Output<Node>& reduction_axes,
                               bool keep_dims)
    : ArithmeticReductionKeepDims(arg, reduction_axes, keep_dims)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceProd::get_default_value() const
{
    return ngraph::make_constant_from_string("1", get_element_type(), get_shape());
}

shared_ptr<Node> op::v1::ReduceProd::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_ReduceProd_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReduceProd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

namespace reduce_prod
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg,
                  const HostTensorPtr& out,
                  const AxisSet& axes,
                  bool keep_dims)
    {
        out->set_shape(reduce(arg->get_shape(), axes, keep_dims));
        runtime::reference::product(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes, keep_dims);
        return true;
    }

    bool evaluate_product(const HostTensorPtr& arg,
                          const HostTensorPtr& out,
                          const AxisSet& axes,
                          bool keep_dims)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_product, i32, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_product, i64, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_product, u32, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_product, u64, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_product, f16, arg, out, axes, keep_dims);
            NGRAPH_TYPE_CASE(evaluate_product, f32, arg, out, axes, keep_dims);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v1::ReduceProd::evaluate(const HostTensorVector& outputs,
                                  const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_ReduceProd_evaluate);
    return reduce_prod::evaluate_product(
        inputs[0], outputs[0], get_reduction_axes(), get_keep_dims());
}

bool op::v1::ReduceProd::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;
    HostTensorPtr lb = input_value(0).get_tensor().get_lower_value(),
                  ub = input_value(0).get_tensor().get_upper_value();
    if (!lb || !ub || !host_tensor_is_positive(lb) || !host_tensor_is_positive(ub))
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v1::ReduceProd::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;
    HostTensorPtr lb = input_value(0).get_tensor().get_lower_value(),
                  ub = input_value(0).get_tensor().get_upper_value();
    if (!lb || !ub || !host_tensor_is_positive(lb) || !host_tensor_is_positive(ub))
        return false;
    return default_upper_bound_evaluator(this, output_values);
}
