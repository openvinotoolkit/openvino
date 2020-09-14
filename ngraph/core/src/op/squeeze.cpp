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
#include <algorithm>
#include <cstddef>
#include <functional>
#include <set>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::Squeeze::type_info;

op::Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes)
    : Op({data, axes})
{
    constructor_validate_and_infer_types();
}

op::Squeeze::Squeeze(const Output<Node>& data)
    : Op({data})
{
    constructor_validate_and_infer_types();
}

void calculate_squeeze_output_based_on_axes(std::vector<int64_t>& axes,
                                            std::vector<Dimension>& output_partial_shape)
{
    const auto& data_rank = output_partial_shape.size();
    std::transform(axes.begin(), axes.end(), axes.begin(), [&data_rank](int64_t axis) -> int64_t {
        const auto& normalized_axis = axis < 0 ? data_rank + axis : axis;
        NGRAPH_CHECK(
            normalized_axis >= 0 && normalized_axis < data_rank, "Axis is out of bounds: ", axis);
        return normalized_axis;
    });
    const auto data_partial_shape = output_partial_shape;
    int idx = 0;
    output_partial_shape.erase(
        std::remove_if(
            output_partial_shape.begin(),
            output_partial_shape.end(),
            [&axes, &idx, &data_partial_shape](const Dimension& d) {
                bool should_remove = std::find(axes.begin(), axes.end(), idx) != axes.end();
                if (should_remove)
                    NGRAPH_CHECK(
                        (data_partial_shape[idx].is_dynamic() || data_partial_shape[idx] == 1),
                        "provided axis value is invalid. Only axes of size 1 may be removed.");
                ++idx;
                return should_remove;
            }),
        output_partial_shape.end());
}

void op::Squeeze::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    const auto& data_partial_shape = input_value(0).get_partial_shape();

    if (data_partial_shape.rank().is_dynamic())
        return;
    bool auto_squeezing = get_input_size() == 1;
    if (!auto_squeezing)
    {
        if (const auto& axes = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
            auto_squeezing = axes->cast_vector<int64_t>().empty();
        else
            return;
    }
   if (auto_squeezing && data_partial_shape.is_dynamic())
        return;

    if (auto_squeezing)
    {
        auto out_shape = data_partial_shape.to_shape();
        out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
        set_output_type(0, get_input_element_type(0), out_shape);
    }
    else
    {
        const auto& axes_constant = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
        auto axes = axes_constant->cast_vector<int64_t>();
        auto output_partial_shape = std::vector<ngraph::Dimension>(data_partial_shape);
        calculate_squeeze_output_based_on_axes(axes, output_partial_shape);
        set_output_type(0, get_input_element_type(0), PartialShape(output_partial_shape));
    }
}

bool ngraph::op::v0::Squeeze::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::Squeeze::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() == 2)
        return make_shared<op::v0::Squeeze>(new_args.at(0), new_args.at(1));
    else if (new_args.size() == 1)
        return make_shared<op::v0::Squeeze>(new_args.at(0));
    else
        throw ngraph_error("Incorrect number of new arguments");
}

namespace
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out)
    {
        runtime::reference::copy(
            arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(out->get_shape()));
        return true;
    }

    bool evaluate_squeeze(const HostTensorPtr& arg0,
                          const HostTensorPtr& arg1,
                          const HostTensorPtr& out)
    {
        auto element_type = arg0->get_element_type();
        out->set_element_type(element_type);

        auto out_shape = std::vector<ngraph::Dimension>(arg0->get_partial_shape());
        if (arg1->get_element_count() == 0)
        {
            out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
        }
        else
        {
            auto axes_shape = arg1->get_shape();
            NGRAPH_CHECK(axes_shape.size() <= 1, "Axes to remove must be a 0/1D vector.");
            vector<int64_t> axes = read_index_vector(arg1);
            calculate_squeeze_output_based_on_axes(axes, out_shape);
        }
        out->set_shape(PartialShape(out_shape).to_shape());

        bool rc = true;
        switch (element_type)
        {
            TYPE_CASE(i32)(arg0, out);
            break;
            TYPE_CASE(i64)(arg0, out);
            break;
            TYPE_CASE(u32)(arg0, out);
            break;
            TYPE_CASE(u64)(arg0, out);
            break;
            TYPE_CASE(f16)(arg0, out);
            break;
            TYPE_CASE(f32)(arg0, out);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Squeeze::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v0::Squeeze::evaluate");
    if (inputs.size() == 2)
    {
        NGRAPH_CHECK(inputs[1], "HostTensor of Squeeze second input (axes) is invalid");
        return evaluate_squeeze(inputs[0], inputs[1], outputs[0]);
    }
    else
        return evaluate_squeeze(inputs[0], HostTensorPtr(), outputs[0]);
}
