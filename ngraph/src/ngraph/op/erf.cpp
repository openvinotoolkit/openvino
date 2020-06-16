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

#include "ngraph/op/erf.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/erf.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Erf::type_info;

bool ngraph::op::v0::Erf::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::Erf::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Erf>(new_args.at(0));
}

op::Erf::Erf(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

#ifdef NGRAPH_EVALUATE_ENABLE
namespace
{
    template <element::Type_t ET>
    inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::erf<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
        return true;
    }

    bool evaluate_erf(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count)
    {
        bool rc = true;
        out->set_unary(arg0);

        switch (arg0->get_element_type())
        {
            TYPE_CASE(boolean)(arg0, out, count);
            break;
            TYPE_CASE(i8)(arg0, out, count);
            break;
            TYPE_CASE(i16)(arg0, out, count);
            break;
            TYPE_CASE(i32)(arg0, out, count);
            break;
            TYPE_CASE(i64)(arg0, out, count);
            break;
            TYPE_CASE(u8)(arg0, out, count);
            break;
            TYPE_CASE(u16)(arg0, out, count);
            break;
            TYPE_CASE(u32)(arg0, out, count);
            break;
            TYPE_CASE(u64)(arg0, out, count);
            break;
            TYPE_CASE(bf16)(arg0, out, count);
            break;
            TYPE_CASE(f16)(arg0, out, count);
            break;
            TYPE_CASE(f32)(arg0, out, count);
            break;
            TYPE_CASE(f64)(arg0, out, count);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::Erf::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_erf(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}
#endif
