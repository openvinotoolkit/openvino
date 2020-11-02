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

#include "constant_folding.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/scatter_elements_update.hpp"
#include "ngraph/runtime/reference/scatter_elements_update.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

template <typename DataType, typename IndicesType, typename AxisType>
static shared_ptr<op::Constant>
    fold_constant_scatter_elem_updt(const shared_ptr<op::Constant>& data,
                                    const shared_ptr<op::Constant>& indices,
                                    const shared_ptr<op::Constant>& updates,
                                    const shared_ptr<op::Constant>& axis,
                                    const shared_ptr<Node>& scatter)
{
    runtime::AlignedBuffer buffer(shape_size(scatter->get_shape()) * sizeof(DataType));
    DataType* data_ptr = buffer.get_ptr<DataType>();

    if (is_type<op::v3::ScatterElementsUpdate>(scatter))
    {
        int64_t normalized_axis = normalize_axis(scatter.get(),
                                                 *(axis->get_data_ptr<AxisType>()),
                                                 static_cast<int64_t>(data->get_shape().size()));

        runtime::reference::scatter_elem_update<DataType, IndicesType>(
            data->get_data_ptr<DataType>(),
            indices->get_data_ptr<IndicesType>(),
            updates->get_data_ptr<DataType>(),
            normalized_axis,
            data_ptr,
            data->get_shape(),
            indices->get_shape());
    }
    else
    {
        throw ngraph_error("Unsupported op in scatter_elem_updt constant folding.");
    }

    return make_shared<op::Constant>(
        scatter->get_output_element_type(0), scatter->get_output_shape(0), data_ptr);
}

template <typename T, typename U>
static shared_ptr<op::Constant>
    dispatch_const_fold_indices(const shared_ptr<op::Constant>& data,
                                const shared_ptr<op::Constant>& indices,
                                const shared_ptr<op::Constant>& updates,
                                const shared_ptr<op::Constant>& axis,
                                const shared_ptr<Node>& scatter_elem_updt)
{
    auto axis_type = axis->get_output_element_type(0);

    // Dispatch specialization based on axis data type.
    switch (axis_type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false,
                     "Encountered 'undefined' element type in constant_scatter_elem_updt_callback");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false,
                     "Encountered 'dynamic' element type in constant_scatter_elem_updt_callback");
        break;
    case element::Type_t::u8:
    case element::Type_t::i8:
        return fold_constant_scatter_elem_updt<T, U, uint8_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::u16:
    case element::Type_t::i16:
        return fold_constant_scatter_elem_updt<T, U, uint16_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::u32:
    case element::Type_t::i32:
        return fold_constant_scatter_elem_updt<T, U, uint32_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::u64:
    case element::Type_t::i64:
        return fold_constant_scatter_elem_updt<T, U, uint64_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::boolean:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
    case element::Type_t::u1:
    default: break;
    }

    NGRAPH_CHECK(
        false,
        "Encountered unsupported axis element type in constant_scatter_elem_updt_callback: ",
        axis_type);
}

template <typename T>
static shared_ptr<op::Constant> dispatch_const_fold_data(const shared_ptr<op::Constant>& data,
                                                         const shared_ptr<op::Constant>& indices,
                                                         const shared_ptr<op::Constant>& updates,
                                                         const shared_ptr<op::Constant>& axis,
                                                         const shared_ptr<Node>& scatter_elem_updt)
{
    auto indices_type = indices->get_output_element_type(0);

    // Dispatch specialization based on indicies data type.
    switch (indices_type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false,
                     "Encountered 'undefined' element type in constant_scatter_elem_updt_callback");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false,
                     "Encountered 'dynamic' element type in constant_scatter_elem_updt_callback");
        break;
    case element::Type_t::u8:
    case element::Type_t::i8:
        return dispatch_const_fold_indices<T, uint8_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::u16:
    case element::Type_t::i16:
        return dispatch_const_fold_indices<T, uint16_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::u32:
    case element::Type_t::i32:
        return dispatch_const_fold_indices<T, uint32_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::u64:
    case element::Type_t::i64:
        return dispatch_const_fold_indices<T, uint64_t>(
            data, indices, updates, axis, scatter_elem_updt);
    case element::Type_t::boolean:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
    case element::Type_t::u1:
    default: break;
    }

    NGRAPH_CHECK(
        false,
        "Encountered unsupported indices element type in constant_scatter_elem_updt_callback: ",
        indices_type);
}

void pass::ConstantFolding::construct_constant_scatter_elements_update()
{
    const auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{10, 20, 30}, pattern::has_class<op::Constant>());
    const auto indices_label = make_shared<pattern::op::Label>(
        element::i64, Shape{5, 10, 15}, pattern::has_class<op::Constant>());
    const auto updates_label = make_shared<pattern::op::Label>(
        element::f32, Shape{5, 10, 15}, pattern::has_class<op::Constant>());
    const auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto scatter_elem_updt = make_shared<op::v3::ScatterElementsUpdate>(
        data_label, indices_label, updates_label, axis_label);

    auto constant_scatter_elem_updt_callback = [this,
                                                data_label,
                                                indices_label,
                                                updates_label,
                                                axis_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_scatter_elem_updt_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        const auto data = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        const auto indices = static_pointer_cast<op::Constant>(pattern_map[indices_label]);
        const auto updates = static_pointer_cast<op::Constant>(pattern_map[updates_label]);
        const auto axis = static_pointer_cast<op::Constant>(pattern_map[axis_label]);
        const auto scatter_elem_updt = m.get_match_root();

        if (cf_is_disabled(scatter_elem_updt))
            return false;

        NGRAPH_CHECK(revalidate_and_ensure_static(scatter_elem_updt));

        std::shared_ptr<Node> replacement;
        const auto data_type = data->get_output_element_type(0);
        NGRAPH_CHECK(data_type == updates->get_output_element_type(0),
                     "data input and updates element type must be equal. Got data type: ",
                     data_type,
                     ", updates type: ",
                     updates->get_output_element_type(0));

        // Dispatch specialization based on data and updates type
        switch (data_type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(
                false,
                "Encountered 'undefined' element type in constant_scatter_elem_updt_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(
                false, "Encountered 'dynamic' element type in constant_scatter_elem_updt_callback");
            break;
        case element::Type_t::boolean:
            NGRAPH_CHECK(
                false, "Encountered 'boolean' element type in constant_scatter_elem_updt_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false,
                         "Encountered 'u1' element type in constant_scatter_elem_updt_callback");
            break;
        case element::Type_t::bf16:
        case element::Type_t::f16:
            replacement =
                dispatch_const_fold_data<float16>(data, indices, updates, axis, scatter_elem_updt);
            break;
        case element::Type_t::f32:
            replacement =
                dispatch_const_fold_data<float>(data, indices, updates, axis, scatter_elem_updt);
            break;
        case element::Type_t::f64:
            replacement =
                dispatch_const_fold_data<double>(data, indices, updates, axis, scatter_elem_updt);
            break;
        case element::Type_t::u8:
        case element::Type_t::i8:
            replacement =
                dispatch_const_fold_data<uint8_t>(data, indices, updates, axis, scatter_elem_updt);
            break;
        case element::Type_t::u16:
        case element::Type_t::i16:
            replacement =
                dispatch_const_fold_data<uint16_t>(data, indices, updates, axis, scatter_elem_updt);
            break;
        case element::Type_t::u32:
        case element::Type_t::i32:
            replacement =
                dispatch_const_fold_data<uint32_t>(data, indices, updates, axis, scatter_elem_updt);
            break;
        case element::Type_t::u64:
        case element::Type_t::i64:
            replacement =
                dispatch_const_fold_data<uint64_t>(data, indices, updates, axis, scatter_elem_updt);
            break;
        default:
            NGRAPH_CHECK(
                false, "Encountered unhandled element type in constant_scatter_elem_updt_callback");
            break;
        }

        replacement->set_friendly_name(m.get_match_root()->get_friendly_name());
        replace_node(m.get_match_root(), replacement);
        copy_runtime_info_to_target_inputs(m.get_match_root(), replacement);
        return true;
    };

    auto scatter_elem_updt_matcher = make_shared<pattern::Matcher>(
        scatter_elem_updt, "ConstantFolding.ConstantScatterElementsUpdateV3");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(scatter_elem_updt_matcher,
                      constant_scatter_elem_updt_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
