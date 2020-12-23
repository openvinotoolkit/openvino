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

#include "itt.hpp"
#include "ngraph/op/atanh.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/atanh.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::Atanh::type_info;

op::v3::Atanh::Atanh(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v3::Atanh::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Atanh>(new_args.at(0));
}

namespace atanhop
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        runtime::reference::atanh(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(arg0->get_shape()));
        return true;
    }

    bool evaluate_atanh(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        bool rc = true;
        out->set_unary(arg0);
        switch (arg0->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_atanh, i32, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_atanh, i64, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_atanh, u32, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_atanh, u64, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_atanh, f16, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_atanh, f32, arg0, out);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v3::Atanh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    bool rc = false;
    NGRAPH_OP_SCOPE(v3_Atanh_evaluate) { rc = atanhop::evaluate_atanh(inputs[0], outputs[0]); }
    return rc;
}
