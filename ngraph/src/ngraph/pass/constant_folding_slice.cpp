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
#include "ngraph/op/slice.hpp"
#include "ngraph/runtime/reference/slice.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_slice(shared_ptr<op::Constant> constant,
                                             shared_ptr<op::Slice> slice)
{
    const Shape& out_shape = slice->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();

    runtime::reference::slice<T>(constant->get_data_ptr<T>(),
                                 data_ptr,
                                 constant->get_shape(),
                                 slice->get_lower_bounds(),
                                 slice->get_upper_bounds(),
                                 slice->get_strides(),
                                 out_shape);

    return make_shared<op::Constant>(constant->get_element_type(), out_shape, data_ptr);
}

void pass::ConstantFolding::construct_constant_slice()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto slice_op = make_shared<op::Slice>(
        data_label, Coordinate{1, 1, 1}, Coordinate{2, 3, 4}, Strides{1, 1, 2});

    auto constant_slice_callback = [data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_slice_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        auto slice = static_pointer_cast<op::Slice>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(slice));

        std::shared_ptr<op::Constant> replacement;

        switch (slice->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_slice");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_slice");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_slice");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_slice<char>(data_node, slice);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_slice<bfloat16>(data_node, slice);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_slice<float16>(data_node, slice);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_slice<float>(data_node, slice);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_slice<double>(data_node, slice);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_slice<int8_t>(data_node, slice);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_slice<int16_t>(data_node, slice);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_slice<int32_t>(data_node, slice);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_slice<int64_t>(data_node, slice);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_slice<uint8_t>(data_node, slice);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_slice<uint16_t>(data_node, slice);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_slice<uint32_t>(data_node, slice);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_slice<uint64_t>(data_node, slice);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto slice_matcher = make_shared<pattern::Matcher>(slice_op, "ConstantFolding.ConstantSlice");
    this->add_matcher(slice_matcher, constant_slice_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
