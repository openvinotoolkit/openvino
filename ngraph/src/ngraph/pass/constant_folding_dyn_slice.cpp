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
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_dyn_slice(shared_ptr<op::Constant> data,
                                                 shared_ptr<op::Constant> lb,
                                                 shared_ptr<op::Constant> ub,
                                                 shared_ptr<op::Constant> strides,
                                                 shared_ptr<op::DynSlice> slice)
{
    SlicePlan plan = make_slice_plan(data->get_shape(),
                                     lb->get_vector<int64_t>(),
                                     ub->get_vector<int64_t>(),
                                     strides->get_vector<int64_t>(),
                                     slice->get_lower_bounds_mask(),
                                     slice->get_upper_bounds_mask(),
                                     slice->get_new_axis(),
                                     slice->get_shrink_axis(),
                                     slice->get_ellipsis_mask());

    runtime::AlignedBuffer slice_out_buffer(shape_size(plan.reshape_in_shape) * sizeof(T));
    T* slice_out_data = slice_out_buffer.get_ptr<T>();
    runtime::reference::slice<T>(data->get_data_ptr<T>(),
                                 slice_out_data,
                                 data->get_shape(),
                                 Coordinate(plan.begins.begin(), plan.begins.end()),
                                 Coordinate(plan.ends.begin(), plan.ends.end()),
                                 Strides(plan.strides.begin(), plan.strides.end()),
                                 plan.reshape_in_shape);

    runtime::AlignedBuffer reshape_out_buffer(shape_size(plan.reshape_out_shape) * sizeof(T));
    T* reshape_out_data = reshape_out_buffer.get_ptr<T>();
    runtime::reference::reshape<T>(slice_out_data,
                                   reshape_out_data,
                                   plan.reshape_in_shape,
                                   get_default_order(plan.reshape_in_shape.size()),
                                   plan.reshape_out_shape);

    runtime::AlignedBuffer reverse_out_buffer(shape_size(plan.reshape_out_shape) * sizeof(T));
    T* reverse_out_data = reverse_out_buffer.get_ptr<T>();
    runtime::reference::reverse<T>(reshape_out_data,
                                   reverse_out_data,
                                   plan.reshape_out_shape,
                                   plan.reshape_out_shape,
                                   plan.reverse_axes);

    return make_shared<op::Constant>(
        data->get_element_type(), plan.reshape_out_shape, reverse_out_data);
}

void pass::ConstantFolding::construct_constant_dyn_slice()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto lb_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto ub_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto strides_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto dyn_slice_op = make_shared<op::DynSlice>(data_label,
                                                  lb_label,
                                                  ub_label,
                                                  strides_label,
                                                  AxisSet{},
                                                  AxisSet{},
                                                  AxisSet{},
                                                  AxisSet{},
                                                  AxisSet{});

    auto constant_dyn_slice_callback = [data_label, lb_label, ub_label, strides_label](
        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dyn_slice_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto lb_node = static_pointer_cast<op::Constant>(pattern_map[lb_label]);
        auto ub_node = static_pointer_cast<op::Constant>(pattern_map[ub_label]);
        auto strides_node = static_pointer_cast<op::Constant>(pattern_map[strides_label]);
        auto dyn_slice = static_pointer_cast<op::DynSlice>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(dyn_slice));

        std::shared_ptr<op::Constant> replacement;

        switch (dyn_slice->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_dyn_slice");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_dyn_slice");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_dyn_slice");
            break;
        case element::Type_t::boolean:
            replacement =
                fold_constant_dyn_slice<char>(data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_dyn_slice<bfloat16>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_dyn_slice<float16>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_dyn_slice<float>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_dyn_slice<double>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_dyn_slice<int8_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_dyn_slice<int16_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_dyn_slice<int32_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_dyn_slice<int64_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_dyn_slice<uint8_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_dyn_slice<uint16_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_dyn_slice<uint32_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_dyn_slice<uint64_t>(
                data_node, lb_node, ub_node, strides_node, dyn_slice);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto dyn_slice_matcher =
        make_shared<pattern::Matcher>(dyn_slice_op, "ConstantFolding.ConstantDynSlice");
    this->add_matcher(
        dyn_slice_matcher, constant_dyn_slice_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
