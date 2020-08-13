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
#include <cstddef>
#include <functional>
#include <set>

#include "ngraph/itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/unsqueeze.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

constexpr NodeTypeInfo op::Unsqueeze::type_info;

op::Unsqueeze::Unsqueeze(const Output<Node>& data, const Output<Node>& axes)
    : FusedOp({data, axes})
{
    constructor_validate_and_infer_types();
}

void op::Unsqueeze::pre_validate_and_infer_types()
{
    const auto data = input_value(0);
    auto data_partial_shape = data.get_partial_shape();
    const auto data_rank = data_partial_shape.rank();

    const auto axes_node = input_value(1).get_node_shared_ptr();

    if (data_rank.is_dynamic() || !op::is_constant(axes_node))
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    uint64_t data_rank_value = data_partial_shape.rank().get_length();

    // Get value of axes from Constant
    const auto axes_constant = as_type_ptr<op::v0::Constant>(axes_node);
    const auto axes_values = axes_constant->cast_vector<int64_t>();
    const auto expanded_rank = data_rank_value + axes_values.size();
    auto axes = normalize_axes(this->description(), axes_values, expanded_rank);

    NODE_VALIDATION_CHECK(this, !axes.empty(), "'axes' input is mandatory.");
    NODE_VALIDATION_CHECK(this,
                          axes.size() == set<int64_t>(begin(axes), end(axes)).size(),
                          "'axes' input has a duplicate axis.");

    sort(begin(axes), end(axes), less<int64_t>());

    vector<Dimension> output_shape{data_partial_shape};
    for (auto axis : axes)
    {
        NODE_VALIDATION_CHECK(
            this, axis <= expanded_rank, "provided 'axes' value ", axis, " is not valid.");

        output_shape.insert(next(begin(output_shape), axis), 1);
    }
    set_output_type(0, get_input_element_type(0), PartialShape{output_shape});
}

OutputVector op::Unsqueeze::decompose_op() const
{
    NODE_VALIDATION_CHECK(
        this,
        (get_output_partial_shape(0).is_static()),
        "output shape was not calculated during pre_validate_and_infer_types. Can not decompose.");
    auto data = input_value(0);
    auto data_shape = data.get_shape();
    auto output_shape = get_output_shape(0);
    AxisVector input_order{ngraph::get_default_order(data_shape.size())};
    return {make_shared<ngraph::op::Reshape>(data, input_order, output_shape)};
}

bool ngraph::op::v0::Unsqueeze::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

shared_ptr<Node> op::Unsqueeze::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Unsqueeze>(new_args.at(0), new_args.at(1));
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

    bool evaluate_unsqueeze(const HostTensorPtr& arg0,
                            const HostTensorPtr& arg1,
                            const HostTensorPtr& out)
    {
        auto element_type = arg0->get_element_type();
        out->set_element_type(element_type);

        auto data_shape = arg0->get_shape();
        int64_t data_rank = static_cast<int64_t>(data_shape.size());
        auto axes_shape = arg1->get_shape();
        NGRAPH_CHECK(axes_shape.size() == 1, "Axes to add must be a vector.");
        NGRAPH_CHECK(axes_shape[0] > 0, "Axes cannot be empty.");

        auto out_shape = data_shape;
        int64_t out_rank = data_rank + static_cast<int64_t>(shape_size(axes_shape));
        // Get axes
        vector<int64_t> axes = read_index_vector(arg1);
        // Normalize axes
        std::transform(axes.begin(), axes.end(), axes.begin(), [out_rank](int64_t i) -> int64_t {
            return i < 0 ? out_rank + i : i;
        });
        // Sort in increasing order
        std::set<int64_t, less<int64_t>> axes_set(axes.begin(), axes.end());
        NGRAPH_CHECK(axes.size() == axes_set.size(), "Axes has duplicate axis.");
        for (int64_t axis : axes_set)
        {
            NGRAPH_CHECK(axis >= 0 && axis < out_rank, "Axis is out of bounds: ", axis);
            out_shape.insert(out_shape.begin() + axis, 1);
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

bool op::v0::Unsqueeze::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::v0::Unsqueeze::evaluate");
    return evaluate_unsqueeze(inputs[0], inputs[1], outputs[0]);
}
