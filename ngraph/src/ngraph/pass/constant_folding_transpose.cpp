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
#include "ngraph/op/transpose.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_transpose(shared_ptr<op::Constant> constant_data,
                                                 shared_ptr<op::Constant> constant_perm,
                                                 shared_ptr<op::Transpose> transpose)
{
    const Shape& out_shape = transpose->get_shape();
    auto input_order = constant_perm->get_axis_vector_val();

    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));

    runtime::opt_kernel::reshape<T>(constant_data->get_data_ptr<T>(),
                                    buffer.get_ptr<T>(),
                                    constant_data->get_shape(),
                                    input_order,
                                    out_shape);

    return make_shared<op::Constant>(transpose->get_element_type(), out_shape, buffer.get_ptr<T>());
}

void pass::ConstantFolding::construct_constant_transpose()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto constant_perm_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto transpose = make_shared<op::Transpose>(constant_data_label, constant_perm_label);

    auto constant_transpose_callback = [constant_data_label,
                                        constant_perm_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_transpose_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_data_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto constant_perm_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_perm_label]);
        auto transpose_match = static_pointer_cast<op::Transpose>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(transpose_match));

        std::shared_ptr<Node> replacement;
        auto type = transpose_match->get_element_type();
        switch (type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in constant_transpose_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false,
                         "Encountered 'dynamic' element type in constant_transpose_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_transpose_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_transpose<char>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_transpose<bfloat16>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_transpose<float16>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_transpose<float>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_transpose<double>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_transpose<int8_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_transpose<int16_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_transpose<int32_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_transpose<int64_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_transpose<uint8_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_transpose<uint16_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_transpose<uint32_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_transpose<uint64_t>(
                constant_data_match, constant_perm_match, transpose_match);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto transpose_matcher =
        make_shared<pattern::Matcher>(transpose, "ConstantFolding.ConstantTranspose");
    this->add_matcher(
        transpose_matcher, constant_transpose_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
