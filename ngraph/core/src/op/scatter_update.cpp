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

#include "ngraph/op/scatter_update.hpp"
#include "ngraph/runtime/reference/scatter_update.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::ScatterUpdate::type_info;

op::v3::ScatterUpdate::ScatterUpdate(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& updates,
                                     const Output<Node>& axis)
    : util::ScatterBase(data, indices, updates, axis)
{
}

shared_ptr<Node> op::v3::ScatterUpdate::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v3::ScatterUpdate>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool op::v3::ScatterUpdate::evaluate(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const auto& axis = inputs[3];
    const auto& out = outputs[0];

    const auto elem_size = data->get_element_type().size();
    out->set_shape(data->get_shape());

    int64_t axis_val = 0;
    switch (axis->get_element_type())
    {
    case element::Type_t::i8: axis_val = axis->get_data_ptr<element::Type_t::i8>()[0]; break;
    case element::Type_t::i16: axis_val = axis->get_data_ptr<element::Type_t::i16>()[0]; break;
    case element::Type_t::i32: axis_val = axis->get_data_ptr<element::Type_t::i32>()[0]; break;
    case element::Type_t::i64: axis_val = axis->get_data_ptr<element::Type_t::i64>()[0]; break;
    case element::Type_t::u8: axis_val = axis->get_data_ptr<element::Type_t::u8>()[0]; break;
    case element::Type_t::u16: axis_val = axis->get_data_ptr<element::Type_t::u16>()[0]; break;
    case element::Type_t::u32: axis_val = axis->get_data_ptr<element::Type_t::u32>()[0]; break;
    case element::Type_t::u64: axis_val = axis->get_data_ptr<element::Type_t::u64>()[0]; break;
    default: throw ngraph_error("axis element type is not integral data type");
    }

    if (axis_val < 0)
    {
        axis_val =
            ngraph::normalize_axis(this, axis_val, static_cast<int64_t>(data->get_shape().size()));
    }

    std::vector<int64_t> indices_casted_vector;
    switch (indices->get_element_type())
    {
    case element::Type_t::i8:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::i8>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::i16:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::i16>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::i32:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::i32>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::i64:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::i64>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::u8:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::u8>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::u16:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::u16>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::u32:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::u32>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    case element::Type_t::u64:
    {
        auto indices_ptr = indices->get_data_ptr<element::Type_t::u64>();
        indices_casted_vector =
            std::vector<int64_t>(indices_ptr, indices_ptr + indices->get_element_count());
        break;
    }
    default: throw ngraph_error("indices element type is not integral data type");
    }

    runtime::reference::scatter_update(data->get_data_ptr<char>(),
                                       indices_casted_vector.data(),
                                       updates->get_data_ptr<char>(),
                                       axis_val,
                                       out->get_data_ptr<char>(),
                                       elem_size,
                                       data->get_shape(),
                                       indices->get_shape(),
                                       updates->get_shape());

    return true;
}
