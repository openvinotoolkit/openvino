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
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/split.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/validation_util.hpp"

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

void pass::ConstantFolding::construct_constant_split()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto split_pattern = make_shared<op::v1::Split>(data_label, axis_label, 0);

    auto constant_split_callback = [this, data_label, axis_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_split_callback against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        const auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        const auto axis_node = static_pointer_cast<op::Constant>(pattern_map[axis_label]);
        const auto split = static_pointer_cast<op::v1::Split>(m.get_match_root());

        const auto axis_val = axis_node->cast_vector<int64_t>()[0];
        const auto norm_axis_val = ngraph::normalize_axis(
            split.get(), axis_val, data_node->get_output_partial_shape(0).rank());
        const auto slices = builder::split(data_node, split->get_num_splits(), norm_axis_val);

        int index = 0;
        for (auto& output : split->outputs())
        {
            output.replace(slices[index++]);
        }
        split->outputs().clear();

        for (auto& slice : as_node_vector(slices))
        {
            auto const_data = std::dynamic_pointer_cast<op::Constant>(
                slice->input_value(0).get_node_shared_ptr());
            auto slice_node = std::dynamic_pointer_cast<op::Slice>(slice);
            if (!const_data || !slice_node)
                continue;

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
                replacement = fold_constant_slice<char>(const_data, slice_node);
                break;
            case element::Type_t::bf16:
                replacement = fold_constant_slice<bfloat16>(const_data, slice_node);
                break;
            case element::Type_t::f16:
                replacement = fold_constant_slice<float16>(const_data, slice_node);
                break;
            case element::Type_t::f32:
                replacement = fold_constant_slice<float>(const_data, slice_node);
                break;
            case element::Type_t::f64:
                replacement = fold_constant_slice<double>(const_data, slice_node);
                break;
            case element::Type_t::i8:
                replacement = fold_constant_slice<int8_t>(const_data, slice_node);
                break;
            case element::Type_t::i16:
                replacement = fold_constant_slice<int16_t>(const_data, slice_node);
                break;
            case element::Type_t::i32:
                replacement = fold_constant_slice<int32_t>(const_data, slice_node);
                break;
            case element::Type_t::i64:
                replacement = fold_constant_slice<int64_t>(const_data, slice_node);
                break;
            case element::Type_t::u8:
                replacement = fold_constant_slice<uint8_t>(const_data, slice_node);
                break;
            case element::Type_t::u16:
                replacement = fold_constant_slice<uint16_t>(const_data, slice_node);
                break;
            case element::Type_t::u32:
                replacement = fold_constant_slice<uint32_t>(const_data, slice_node);
                break;
            case element::Type_t::u64:
                replacement = fold_constant_slice<uint64_t>(const_data, slice_node);
                break;
            }
            replace_node(slice_node, replacement);
        }

        return true;
    };
    auto split_matcher =
        make_shared<pattern::Matcher>(split_pattern, "ConstantFolding.ConstantSplit");
    this->add_matcher(split_matcher, constant_split_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
