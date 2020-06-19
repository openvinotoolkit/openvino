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

#include "ngraph/op/reduce_prod.hpp"
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
    check_new_args_count(this, new_args);
    return make_shared<ReduceProd>(new_args.at(0), new_args.at(1), get_keep_dims());
}

#ifdef NGRAPH_EVALUATE_ENABLE
namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        std::cout << "AA 116" << std::endl;
        out->set_shape(reduce(arg->get_shape(), axes));
        runtime::reference::product(
            arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), arg->get_shape(), axes);
        return true;
    }

    bool evaluate_product(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        bool rc = true;
        switch (arg->get_element_type())
        {
            TYPE_CASE(i32)(arg, out, axes);
            break;
            TYPE_CASE(i64)(arg, out, axes);
            break;
            TYPE_CASE(u32)(arg, out, axes);
            break;
            TYPE_CASE(u64)(arg, out, axes);
            break;
            TYPE_CASE(f16)(arg, out, axes);
            break;
            TYPE_CASE(f32)(arg, out, axes);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v1::ReduceProd::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    std::cout << "AA 117" << std::endl;
    return evaluate_product(inputs[0], outputs[0], get_reduction_axes());
}
#endif
