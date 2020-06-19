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

#include <string>
#include <vector>

#include "ngraph/op/asinh.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/asinh.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::Asinh::type_info;

op::v3::Asinh::Asinh(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v3::Asinh::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Asinh>(new_args.at(0));
}

#ifdef NGRAPH_EVALUATE_ENABLE
namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        std::cout << "AA 17" << std::endl;
        runtime::reference::asinh(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(arg0->get_shape()));
        return true;
    }

    bool evaluate_asinh(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        bool rc = true;
        out->set_unary(arg0);
        switch (arg0->get_element_type())
        {
            TYPE_CASE(i8)(arg0, out);
            break;
            TYPE_CASE(i16)(arg0, out);
            break;
            TYPE_CASE(i32)(arg0, out);
            break;
            TYPE_CASE(i64)(arg0, out);
            break;
            TYPE_CASE(u8)(arg0, out);
            break;
            TYPE_CASE(u16)(arg0, out);
            break;
            TYPE_CASE(u32)(arg0, out);
            break;
            TYPE_CASE(u64)(arg0, out);
            break;
            TYPE_CASE(bf16)(arg0, out);
            break;
            TYPE_CASE(f16)(arg0, out);
            break;
            TYPE_CASE(f32)(arg0, out);
            break;
            TYPE_CASE(f64)(arg0, out);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v3::Asinh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    std::cout << "AA 18" << std::endl;
    return evaluate_asinh(inputs[0], outputs[0]);
}
#endif
