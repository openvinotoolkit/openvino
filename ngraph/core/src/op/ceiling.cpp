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

#include "ngraph/op/ceiling.hpp"
#include "itt.hpp"
#include "ngraph/op/util/eval_copy.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/copy.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Ceiling::type_info;

op::Ceiling::Ceiling(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Ceiling::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Ceiling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Ceiling>(new_args.at(0));
}

namespace ceiling
{
    // function used by TYPE_CASE
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ceiling<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    // function used by COPY_TENSOR
    template <element::Type_t ET>
    inline bool copy_tensor(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        runtime::reference::copy(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_ceiling(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            NGRAPH_COPY_TENSOR(evaluate_ceiling, boolean, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, i8, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, i16, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, i32, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, i64, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, u8, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, u16, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, u32, arg0, out, count);
            NGRAPH_COPY_TENSOR(evaluate_ceiling, u64, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_ceiling, f16, arg0, out, count);
            NGRAPH_TYPE_CASE(evaluate_ceiling, f32, arg0, out, count);
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Ceiling::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Ceiling_evaluate);
    return ceiling::evaluate_ceiling(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}
