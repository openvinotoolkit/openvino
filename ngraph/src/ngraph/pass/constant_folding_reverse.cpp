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
#include "ngraph/op/reverse.hpp"
#include "ngraph/runtime/reference/reverse.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static shared_ptr<op::Constant> fold_constant_reverse_helper(shared_ptr<op::Constant> constant,
                                                             const AxisSet& reversed_axes)
{
    const Shape& out_shape = constant->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();

    runtime::reference::reverse<T>(
        constant->get_vector<T>().data(), data_ptr, out_shape, out_shape, reversed_axes);

    return make_shared<op::Constant>(constant->get_output_element_type(0), out_shape, data_ptr);
}

static shared_ptr<op::Constant> fold_constant_reverse(shared_ptr<op::Constant> constant,
                                                      const AxisSet& reversed_axes)
{
    auto& input_element_type = constant->get_output_element_type(0);

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (input_element_type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_convert");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_convert");
        break;
    case element::Type_t::u1:
        NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_convert");
        break;
    case element::Type_t::boolean:
        return fold_constant_reverse_helper<char>(constant, reversed_axes);
    case element::Type_t::bf16:
        return fold_constant_reverse_helper<bfloat16>(constant, reversed_axes);
    case element::Type_t::f16:
        return fold_constant_reverse_helper<float16>(constant, reversed_axes);
    case element::Type_t::f32: return fold_constant_reverse_helper<float>(constant, reversed_axes);
    case element::Type_t::f64: return fold_constant_reverse_helper<double>(constant, reversed_axes);
    case element::Type_t::i8: return fold_constant_reverse_helper<int8_t>(constant, reversed_axes);
    case element::Type_t::i16:
        return fold_constant_reverse_helper<int16_t>(constant, reversed_axes);
    case element::Type_t::i32:
        return fold_constant_reverse_helper<int32_t>(constant, reversed_axes);
    case element::Type_t::i64:
        return fold_constant_reverse_helper<int64_t>(constant, reversed_axes);
    case element::Type_t::u8: return fold_constant_reverse_helper<uint8_t>(constant, reversed_axes);
    case element::Type_t::u16:
        return fold_constant_reverse_helper<uint16_t>(constant, reversed_axes);
    case element::Type_t::u32:
        return fold_constant_reverse_helper<uint32_t>(constant, reversed_axes);
    case element::Type_t::u64:
        return fold_constant_reverse_helper<uint64_t>(constant, reversed_axes);
    }

    NGRAPH_UNREACHABLE("Unexpected switch case");

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

void pass::ConstantFolding::construct_constant_reverse()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto convert_op = make_shared<op::Reverse>(constant_label, AxisSet{0, 1, 2});

    auto constant_reverse_callback = [constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reverse_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto reverse_match = static_pointer_cast<op::Reverse>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(reverse_match));

        replace_node(m.get_match_root(),
                     fold_constant_reverse(constant_match, reverse_match->get_reversed_axes()));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantReverse");
    this->add_matcher(
        convert_matcher, constant_reverse_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
