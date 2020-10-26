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

NGRAPH_RTTI_DEFINITION(op::v0::Squeeze, "Squeeze", 0);

op::Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes)
    : FusedOp({data, axes})
{
    constructor_validate_and_infer_types();
}

void op::Squeeze::pre_validate_and_infer_types()
{
    auto data = input_value(0);
    auto axes_node = input_value(1).get_node_shared_ptr();

    bool data_has_dynamic_rank = data.get_partial_shape().rank().is_dynamic();
    bool data_has_dynamic_shape = data.get_partial_shape().is_dynamic();

    auto axes_constant = as_type_ptr<op::v0::Constant>(axes_node);
    bool axes_is_empty_constant =
        (axes_constant) ? axes_constant->cast_vector<int64_t>().empty() : false;

    if (data_has_dynamic_rank || !axes_constant ||
        (data_has_dynamic_shape && axes_is_empty_constant))
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    auto data_partial_shape = data.get_partial_shape();
    uint64_t data_rank = data_partial_shape.rank().get_length();

    // Get value of axes from Constant
    auto axes =
        normalize_axes(this->description(), axes_constant->cast_vector<int64_t>(), data_rank);

    // Prepare set of unique axes marked to be removed from input data.
    vector<uint64_t> axes_to_squeeze(data_rank);
    if (axes_is_empty_constant)
    {
        auto data_shape = data.get_shape();
        // Default behaviour is to remove all single dimension axes.
        for (uint64_t idx = 0; idx < data_rank; ++idx)
        {
            if (data_shape.at(idx) == 1)
            {
                axes_to_squeeze.at(idx) = 1;
            }
            else
            {
                axes_to_squeeze.at(idx) = 0;
            }
        }
    }
    else
    {
        set<size_t, greater<size_t>> unique_axes(begin(axes), end(axes));
        for (uint64_t axis : unique_axes)
        {
            if (!data_has_dynamic_shape)
            {
                auto data_shape = data.get_shape();
                NODE_VALIDATION_CHECK(
                    this,
                    (data_shape.at(axis) == 1),
                    "provided axis value is invalid. Only axes of size 1 may be removed.");
            }
            axes_to_squeeze.at(axis) = 1;
        }
    }

    vector<Dimension> output_data_shape;
    for (uint64_t idx = 0; idx < data_rank; ++idx)
    {
        if (axes_to_squeeze.at(idx) == 0)
        {
            output_data_shape.push_back(data_partial_shape[idx]);
        }
    }
    set_output_type(0, get_input_element_type(0), PartialShape(output_data_shape));
}

bool ngraph::op::v0::Squeeze::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

OutputVector op::Squeeze::decompose_op() const
{
    NODE_VALIDATION_CHECK(
        this,
        (get_output_partial_shape(0).is_static()),
        "output shape was not calculated during pre_validate_and_infer_types. Can not decompose.");
    auto data = input_value(0);
    auto data_shape = data.get_shape();
    auto output_data_shape = get_output_shape(0);
    AxisVector input_order{get_default_order(data_shape.size())};
    return {make_shared<op::Reshape>(data, input_order, output_data_shape)};
}

shared_ptr<Node> op::Squeeze::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Squeeze>(new_args.at(0), new_args.at(1));
}

namespace squeeze
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

        auto data_shape = arg0->get_shape();
        int64_t data_rank = static_cast<int64_t>(data_shape.size());
        auto axes_shape = arg1->get_shape();
        NGRAPH_CHECK(axes_shape.size() <= 1, "Axes to remove must be a vector or empty.");

        auto out_shape = data_shape;
        // Empty axes vector
        if (axes_shape.size() == 0 || axes_shape[0] == 0)
        {
            out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1), out_shape.end());
        }
        else
        {
            // Get axes
            vector<int64_t> axes = read_index_vector(arg1);
            // Normalize axes
            std::transform(axes.begin(),
                           axes.end(),
                           axes.begin(),
                           [data_rank](int64_t i) -> int64_t { return i < 0 ? data_rank + i : i; });
            // Sort in decreasing order
            std::set<int64_t, greater<int64_t>> axes_set(axes.begin(), axes.end());
            for (int64_t axis : axes_set)
            {
                NGRAPH_CHECK(axis >= 0 && axis < data_rank, "Axis is out of bounds: ", axis);
                NGRAPH_CHECK(out_shape[axis] == 1, "Only axis of size 1 can be removed.");
                out_shape.erase(out_shape.begin() + axis);
            }
        }
        out->set_shape(out_shape);

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
    return squeeze::evaluate_squeeze(inputs[0], inputs[1], outputs[0]);
}

bool op::v0::Squeeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values)
{
    if (get_output_partial_shape(0).is_dynamic())
    {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const =
            std::dynamic_pointer_cast<op::Constant>(inputs_values[0].get_node_shared_ptr()))
    {
        // In case if data constant has single consumer we can change it shape without making a copy
        // Otherwise we create Constant copy with shape from squeeze node
        if (data_const->output(0).get_target_inputs().size() == 1)
        {
            data_const->set_data_shape(shape);
            data_const->validate_and_infer_types();
            output_values[0] = data_const;
        }
        else
        {
            output_values[0] = std::make_shared<op::Constant>(
                data_const->get_element_type(), shape, data_const->get_data_ptr());
        }
        return true;
    }
    return false;
}