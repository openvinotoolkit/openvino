// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <functional>
#include <set>

#include "itt.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/unsqueeze.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::Unsqueeze, "Unsqueeze", 0);

op::v0::Unsqueeze::Unsqueeze(const Output<Node>& data, const Output<Node>& axes)
    : Op({data, axes})
{
    constructor_validate_and_infer_types();
}

void op::v0::Unsqueeze::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_Unsqueeze_validate_and_infer_types);
    const auto data = input_value(0);
    auto data_partial_shape = data.get_partial_shape();
    const auto data_rank = data_partial_shape.rank();

    const auto axes_constant = get_constant_from_source(input_value(1));
    auto axes_pshape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          axes_pshape.rank().compatible(0) || axes_pshape.rank().compatible(1),
                          "Second input (axes) should not be of rank higher than 1. Got: ",
                          axes_pshape.rank().get_length());

    if (data_rank.is_dynamic() || !axes_constant)
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    const auto axes_values = axes_constant->cast_vector<int64_t>();
    uint64_t data_rank_value = data_partial_shape.rank().get_length();
    const int64_t expanded_rank = data_rank_value + axes_values.size();

    NODE_VALIDATION_CHECK(this, !axes_values.empty(), "'axes' input is mandatory");

    auto normalized_axes = normalize_axes(this->description(), axes_values, expanded_rank);
    set<int64_t> axes(begin(normalized_axes), end(normalized_axes));
    vector<Dimension> output_shape{data_partial_shape};
    for (auto axis : axes)
    {
        NODE_VALIDATION_CHECK(
            this, axis <= expanded_rank, "provided 'axes' value ", axis, " is not valid.");

        output_shape.insert(next(begin(output_shape), axis), 1);
    }
    set_output_type(0, get_input_element_type(0), PartialShape{output_shape});
}

bool op::v0::Unsqueeze::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Unsqueeze_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Unsqueeze::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Unsqueeze_clone_with_new_inputs);
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Unsqueeze>(new_args.at(0), new_args.at(1));
}

namespace unsqueeze
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
            NGRAPH_TYPE_CASE(evaluate_unsqueeze, i32, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_unsqueeze, i64, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_unsqueeze, u32, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_unsqueeze, u64, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_unsqueeze, f16, arg0, out);
            NGRAPH_TYPE_CASE(evaluate_unsqueeze, f32, arg0, out);
        default: rc = false; break;
        }
        return rc;
    }
} // namespace unsqueeze

bool op::v0::Unsqueeze::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Unsqueeze_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 2));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));
    return unsqueeze::evaluate_unsqueeze(inputs[0], inputs[1], outputs[0]);
}

bool op::v0::Unsqueeze::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_Unsqueeze_has_evaluate);
    switch (get_input_element_type(0))
    {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}

bool op::v0::Unsqueeze::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v0::Unsqueeze::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v0::Unsqueeze::constant_fold(OutputVector& output_values,
                                      const OutputVector& inputs_values)
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
        // Otherwise we create Constant copy with shape from unsqueeze node
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
