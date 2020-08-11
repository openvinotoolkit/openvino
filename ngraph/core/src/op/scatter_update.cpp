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

namespace
{
    template <element::Type_t DT, element::Type_t IT>
    bool evaluate(const HostTensorPtr& data,
                  const HostTensorPtr& indices,
                  const HostTensorPtr& updates,
                  const HostTensorPtr& out,
                  const int64_t normalized_axis)
    {
        using DataType = typename element_type_traits<DT>::value_type;
        using IndicesType = typename element_type_traits<IT>::value_type;

        out->set_shape(data->get_shape());
        runtime::reference::scatter_update<DataType, IndicesType>(
            data->get_data_ptr<DataType>(),
            indices->get_data_ptr<IndicesType>(),
            updates->get_data_ptr<DataType>(),
            normalized_axis,
            out->get_data_ptr<DataType>(),
            data->get_shape(),
            indices->get_shape(),
            updates->get_shape());

        return true;
    }

    template <element::Type_t DT>
    bool evaluate(const HostTensorPtr& data,
                  const HostTensorPtr& indices,
                  const HostTensorPtr& updates,
                  const HostTensorPtr& out,
                  const int64_t normalized_axis)
    {
        // Dispatch specialization based on indicies data type.
        bool rc = true;

        switch (indices->get_element_type())
        {
        case element::Type_t::i8:
        case element::Type_t::u8:
            rc = evaluate<DT, element::Type_t::u8>(data, indices, updates, out, normalized_axis);
            break;
        case element::Type_t::i16:
        case element::Type_t::u16:
            rc = evaluate<DT, element::Type_t::u16>(data, indices, updates, out, normalized_axis);
            break;
        case element::Type_t::i32:
        case element::Type_t::u32:
            rc = evaluate<DT, element::Type_t::u32>(data, indices, updates, out, normalized_axis);
            break;
        case element::Type_t::i64:
        case element::Type_t::u64:
            rc = evaluate<DT, element::Type_t::u64>(data, indices, updates, out, normalized_axis);
            break;
        default: rc = false; break;
        }
        return rc;
    }

    bool evaluate_scatter_update(const HostTensorPtr& data,
                                 const HostTensorPtr& indices,
                                 const HostTensorPtr& updates,
                                 const HostTensorPtr& out,
                                 const int64_t normalized_axis)
    {
        // Dispatch based on data, updates and output data type.
        bool rc = true;
        switch (out->get_element_type())
        {
        case element::Type_t::i32:
        case element::Type_t::u32:
            rc = evaluate<element::Type_t::u32>(data, indices, updates, out, normalized_axis);
            break;
        case element::Type_t::i64:
        case element::Type_t::u64:
            rc = evaluate<element::Type_t::u64>(data, indices, updates, out, normalized_axis);
            break;
            TYPE_CASE(f16)(data, indices, updates, out, normalized_axis);
            break;
            TYPE_CASE(f32)(data, indices, updates, out, normalized_axis);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v3::ScatterUpdate::evaluate(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    const auto& axis = inputs[3];
    const auto& out = outputs[0];

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

    const auto& input_rank = get_input_partial_shape(0).rank();
    int64_t normalized_axis = axis_val;

    if (normalized_axis < 0)
    {
        normalized_axis =
            ngraph::normalize_axis(this, axis_val, static_cast<int64_t>(data->get_shape().size()));
    }

    return evaluate_scatter_update(data, indices, updates, out, normalized_axis);
}
