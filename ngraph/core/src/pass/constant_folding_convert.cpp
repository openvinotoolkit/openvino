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
#include "ngraph/op/convert.hpp"
#include "ngraph/runtime/reference/convert.hpp"

using namespace std;
using namespace ngraph;

// Helper for mapping element::Types to runtime::reference::convert, which is templated in C++
// data types. Used by fold_constant_convert and fold_constant_convert_helper0, which respectively
// determine the appropriate C++ types for "TI" (input type) and "TO" (output type).
template <typename TI, typename TO>
shared_ptr<op::Constant> fold_constant_convert_helper1(shared_ptr<op::Constant> constant,
                                                       const element::Type& output_element_type)
{
    const Shape& out_shape = constant->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(TO));
    TO* data_ptr = buffer.get_ptr<TO>();

    runtime::reference::convert<TI, TO>(
        constant->get_data_ptr<TI>(), data_ptr, shape_size(out_shape));

    return make_shared<op::Constant>(output_element_type, out_shape, data_ptr);
}

// Helper for mapping element::Types to runtime::reference::convert, which is templated in C++
// data types. Used by fold_constant_convert, which determines the appropriate C++ type for "TI"
// (input type).
template <typename TI>
shared_ptr<op::Constant> fold_constant_convert_helper0(shared_ptr<op::Constant> constant,
                                                       const element::Type& output_element_type)
{
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (output_element_type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_convert");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_convert");
        break;
    case element::Type_t::u1:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_convert");
        break;
    case element::Type_t::boolean:
        return fold_constant_convert_helper1<TI, char>(constant, output_element_type);
    case element::Type_t::bf16:
        return fold_constant_convert_helper1<TI, bfloat16>(constant, output_element_type);
    case element::Type_t::f16:
        return fold_constant_convert_helper1<TI, float16>(constant, output_element_type);
    case element::Type_t::f32:
        return fold_constant_convert_helper1<TI, float>(constant, output_element_type);
    case element::Type_t::f64:
        return fold_constant_convert_helper1<TI, double>(constant, output_element_type);
    case element::Type_t::i8:
        return fold_constant_convert_helper1<TI, int8_t>(constant, output_element_type);
    case element::Type_t::i16:
        return fold_constant_convert_helper1<TI, int16_t>(constant, output_element_type);
    case element::Type_t::i32:
        return fold_constant_convert_helper1<TI, int32_t>(constant, output_element_type);
    case element::Type_t::i64:
        return fold_constant_convert_helper1<TI, int64_t>(constant, output_element_type);
    case element::Type_t::u8:
        return fold_constant_convert_helper1<TI, uint8_t>(constant, output_element_type);
    case element::Type_t::u16:
        return fold_constant_convert_helper1<TI, uint16_t>(constant, output_element_type);
    case element::Type_t::u32:
        return fold_constant_convert_helper1<TI, uint32_t>(constant, output_element_type);
    case element::Type_t::u64:
        return fold_constant_convert_helper1<TI, uint64_t>(constant, output_element_type);
    }

    NGRAPH_UNREACHABLE("Unexpected switch case");
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

static shared_ptr<op::Constant> fold_constant_convert(shared_ptr<op::Constant> constant,
                                                      const element::Type& output_element_type)
{
    auto& input_element_type = constant->get_output_element_type(0);

    if (input_element_type == output_element_type)
    {
        return constant;
    }

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
        return fold_constant_convert_helper0<char>(constant, output_element_type);
    case element::Type_t::bf16:
        return fold_constant_convert_helper0<bfloat16>(constant, output_element_type);
    case element::Type_t::f16:
        return fold_constant_convert_helper0<float16>(constant, output_element_type);
    case element::Type_t::f32:
        return fold_constant_convert_helper0<float>(constant, output_element_type);
    case element::Type_t::f64:
        return fold_constant_convert_helper0<double>(constant, output_element_type);
    case element::Type_t::i8:
        return fold_constant_convert_helper0<int8_t>(constant, output_element_type);
    case element::Type_t::i16:
        return fold_constant_convert_helper0<int16_t>(constant, output_element_type);
    case element::Type_t::i32:
        return fold_constant_convert_helper0<int32_t>(constant, output_element_type);
    case element::Type_t::i64:
        return fold_constant_convert_helper0<int64_t>(constant, output_element_type);
    case element::Type_t::u8:
        return fold_constant_convert_helper0<uint8_t>(constant, output_element_type);
    case element::Type_t::u16:
        return fold_constant_convert_helper0<uint16_t>(constant, output_element_type);
    case element::Type_t::u32:
        return fold_constant_convert_helper0<uint32_t>(constant, output_element_type);
    case element::Type_t::u64:
        return fold_constant_convert_helper0<uint64_t>(constant, output_element_type);
    }

    NGRAPH_UNREACHABLE("Unexpected switch case");
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

void pass::ConstantFolding::construct_constant_convert()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto convert_op = make_shared<op::Convert>(constant_label, element::i64);

    auto constant_convert_callback = [this, constant_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_convert_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_label]);
        auto convert_match = static_pointer_cast<op::Convert>(m.get_match_root());

        if (cf_is_disabled(convert_match))
            return false;

        NGRAPH_CHECK(revalidate_and_ensure_static(convert_match));

        replace_node(
            m.get_match_root(),
            fold_constant_convert(constant_match, convert_match->get_output_element_type(0)));
        return true;
    };

    auto convert_matcher =
        make_shared<pattern::Matcher>(convert_op, "ConstantFolding.ConstantConvert");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(
        convert_matcher, constant_convert_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
