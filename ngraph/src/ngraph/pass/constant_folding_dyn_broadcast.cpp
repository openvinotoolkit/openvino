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
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_dyn_broadcast(shared_ptr<op::Constant> arg,
                                                     shared_ptr<op::Constant> shape,
                                                     shared_ptr<op::Constant> axes)
{
    const Shape& out_shape = shape->get_shape_val();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();

    runtime::reference::broadcast<T>(
        arg->get_data_ptr<T>(), data_ptr, arg->get_shape(), out_shape, axes->get_axis_set_val());

    return make_shared<op::Constant>(arg->get_element_type(), out_shape, data_ptr);
}

void pass::ConstantFolding::construct_constant_dyn_broadcast()
{
    auto constant_arg_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());
    auto constant_shape_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto constant_axes_label =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());

    auto dyn_broadcast = make_shared<op::DynBroadcast>(
        constant_arg_label, constant_shape_label, constant_axes_label);

    auto constant_dyn_broadcast_callback = [constant_arg_label,
                                            constant_shape_label,
                                            constant_axes_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dyn_broadcast_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_arg_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_arg_label]);
        auto constant_shape_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_shape_label]);
        auto constant_axes_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_axes_label]);
        auto dyn_broadcast_match = static_pointer_cast<op::DynBroadcast>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(dyn_broadcast_match));

        std::shared_ptr<Node> replacement;
        auto type = dyn_broadcast_match->get_output_element_type(0);
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_dyn_broadcast_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_dyn_broadcast_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_dyn_broadcast_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_dyn_broadcast<char>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_dyn_broadcast<bfloat16>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_dyn_broadcast<float16>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_dyn_broadcast<float>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_dyn_broadcast<double>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_dyn_broadcast<int8_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_dyn_broadcast<int16_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_dyn_broadcast<int32_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_dyn_broadcast<int64_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_dyn_broadcast<uint8_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_dyn_broadcast<uint16_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_dyn_broadcast<uint32_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_dyn_broadcast<uint64_t>(
                constant_arg_match, constant_shape_match, constant_axes_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto dyn_broadcast_matcher =
        make_shared<pattern::Matcher>(dyn_broadcast, "ConstantFolding.ConstantDynBroadcast");
    this->add_matcher(
        dyn_broadcast_matcher, constant_dyn_broadcast_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
