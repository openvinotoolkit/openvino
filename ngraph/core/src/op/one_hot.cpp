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

#include "ngraph/op/one_hot.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::OneHot::type_info;

op::v1::OneHot::OneHot(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis)
    : Op({indices, depth, on_value, off_value})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

void op::v1::OneHot::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_OneHot_validate_and_infer_types);
    const auto& indices_et = get_input_element_type(0);
    const auto& depth_et = get_input_element_type(1);
    const auto& on_value_et = get_input_element_type(2);
    const auto& off_value_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_dynamic() || indices_et.is_integral(),
                          "Indices must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          depth_et.is_dynamic() || depth_et.is_integral(),
                          "Depth must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          on_value_et.compatible(off_value_et),
                          "on_value element type must be compatible with off_value element type.");

    const auto& indices_shape = get_input_partial_shape(0);
    const auto& depth_shape = get_input_partial_shape(1);
    const auto& on_value_shape = get_input_partial_shape(2);
    const auto& off_value_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          depth_shape.is_dynamic() || is_scalar(depth_shape.to_shape()),
                          "depth input must be scalar.");

    NODE_VALIDATION_CHECK(this,
                          on_value_shape.is_dynamic() || is_scalar(on_value_shape.to_shape()),
                          "on_value input must be scalar.");

    NODE_VALIDATION_CHECK(this,
                          off_value_shape.is_dynamic() || is_scalar(off_value_shape.to_shape()),
                          "off_value input must be scalar.");

    PartialShape result_shape{PartialShape::dynamic()};
    const auto& depth = input_value(1).get_node_shared_ptr();
    const auto& depth_constant = get_constant_from_source(input_value(1));
    if (indices_shape.rank().is_static() && depth_constant)
    {
        std::vector<Dimension> out_dims{indices_shape};
        const auto indices_rank = indices_shape.rank().get_length();
        m_axis =
            ngraph::normalize_axis(this, m_axis, indices_rank + 1, -indices_rank - 1, indices_rank);

        auto depth_element_type = depth->get_output_element_type(0);
        NODE_VALIDATION_CHECK(this,
                              depth_element_type.is_integral(),
                              "'depth' input element type must be an integer (got ",
                              depth_element_type,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              is_scalar(depth->get_shape()),
                              "A scalar input should be provided as 'depth' to OneHot",
                              " (got ",
                              depth->get_shape(),
                              " elements).");

        int64_t depth_val = depth_constant->cast_vector<int64_t>()[0];
        NODE_VALIDATION_CHECK(this,
                              depth_val > 0,
                              "The value of 'depth' must be a positive number.",
                              " (got ",
                              depth_val,
                              ").");
        out_dims.insert(out_dims.begin() + m_axis, Dimension(depth_val));
        result_shape = out_dims;
    }

    set_output_type(0, on_value_et, result_shape);
}

bool ngraph::op::v1::OneHot::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_OneHot_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v1::OneHot::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_OneHot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::OneHot>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_axis);
}

namespace one_hot
{
    template <element::Type_t T>
    bool evaluate(const HostTensorVector& output_values,
                  const HostTensorVector& input_values,
                  const int64_t axis)
    {
        using INPUT_TYPE = typename element_type_traits<T>::value_type;
        const auto& indices = input_values[0];
        const auto& on_value = input_values[2];
        const auto& off_value = input_values[3];
        const auto& out = output_values[0];
        runtime::reference::one_hot<INPUT_TYPE>(indices->get_data_ptr<INPUT_TYPE>(),
                                                indices->get_shape(),
                                                out->get_data_ptr<char>(),
                                                out->get_element_type().size(),
                                                out->get_shape()[axis],
                                                axis,
                                                on_value->get_data_ptr<char>(),
                                                off_value->get_data_ptr<char>());
        return true;
    }
    bool evaluate_onehot(const HostTensorVector& output_values,
                         const HostTensorVector& input_values,
                         const int64_t axis)
    {
        bool rc = true;
        const auto& indices = input_values[0];
        switch (indices->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_onehot, i32, output_values, input_values, axis);
            NGRAPH_TYPE_CASE(evaluate_onehot, i64, output_values, input_values, axis);
        default: rc = false;
        }
        return rc;
    }
} // namespace one_hot

bool op::v1::OneHot::evaluate(const HostTensorVector& output_values,
                              const HostTensorVector& input_values) const
{
    NGRAPH_OP_SCOPE(v1_OneHot_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 4));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));

    const auto& ind_Pshape = input_values[0]->get_partial_shape();
    const auto& out_Pshape = output_values[0]->get_partial_shape();
    NGRAPH_CHECK(ind_Pshape.is_static() && out_Pshape.is_static(),
                 "Only static input/output shapes are supported");
    const auto out_shape = out_Pshape.get_shape();
    const auto axis = get_axis();
    NGRAPH_CHECK(axis >= 0 && axis < out_shape.size(), "Invalid axis value.");
    const auto depth = get_constant_from_source(input_value(1))->cast_vector<int64_t>()[0];
    const auto ind_shape = ind_Pshape.get_shape();
    NGRAPH_CHECK(shape_size(ind_shape) * depth == shape_size(out_shape),
                 "Incompatible I/O shapes or wrong depth value.");
    NGRAPH_CHECK(out_shape[axis] == depth, "Incompatible axis and depth values.");
    return one_hot::evaluate_onehot(output_values, input_values, axis);
}