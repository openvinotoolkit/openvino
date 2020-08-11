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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"

using namespace std;
using namespace ngraph;

template <class INDICES_TYPE, class OUTPUT_TYPE>
shared_ptr<op::Constant> fold_constant_one_hot_ref(const shared_ptr<op::Constant>& indices,
                                                   const shared_ptr<op::Constant>& on_value,
                                                   const shared_ptr<op::Constant>& off_value,
                                                   const Shape& output_shape,
                                                   size_t axis)
{
    std::vector<OUTPUT_TYPE> out_vec(shape_size(output_shape));
    runtime::reference::one_hot<INDICES_TYPE, OUTPUT_TYPE>(
        indices->get_data_ptr<INDICES_TYPE>(),
        out_vec.data(),
        indices->get_shape(),
        output_shape,
        axis,
        on_value->get_data_ptr<OUTPUT_TYPE>()[0],
        off_value->get_data_ptr<OUTPUT_TYPE>()[0]);

    return make_shared<op::Constant>(on_value->get_element_type(), output_shape, out_vec);
}

template <class OUTPUT_TYPE>
shared_ptr<op::Constant> fold_constant_one_hot(const shared_ptr<op::Constant>& indices,
                                               const shared_ptr<op::Constant>& on_value,
                                               const shared_ptr<op::Constant>& off_value,
                                               const Shape& output_shape,
                                               size_t axis)
{
    shared_ptr<op::Constant> rc;
    switch (indices->get_element_type())
    {
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::u1:
    case element::Type_t::boolean:
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::f32:
    case element::Type_t::f64:
        NGRAPH_CHECK(false, "Indices input element type must be integer");
        break;
    case element::Type_t::i8:
        rc = fold_constant_one_hot_ref<int8_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::i16:
        rc = fold_constant_one_hot_ref<int16_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::i32:
        rc = fold_constant_one_hot_ref<int32_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::i64:
        rc = fold_constant_one_hot_ref<int64_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::u8:
        rc = fold_constant_one_hot_ref<uint8_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::u16:
        rc = fold_constant_one_hot_ref<uint16_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::u32:
        rc = fold_constant_one_hot_ref<uint32_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    case element::Type_t::u64:
        rc = fold_constant_one_hot_ref<uint64_t, OUTPUT_TYPE>(
            indices, on_value, off_value, output_shape, axis);
        break;
    default: NGRAPH_CHECK(false, "Indices input element type must be integer");
    }
    return rc;
}

void pass::ConstantFolding::construct_constant_one_hot()
{
    auto indices_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto depth_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto on_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto off_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    int64_t axis = 0;
    auto ont_hot_pattern =
        make_shared<op::v1::OneHot>(indices_label, depth_label, on_label, off_label, axis);

    auto one_hot_callback = [indices_label, depth_label, on_label, off_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for one_hot_callback against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto indices_node = static_pointer_cast<op::Constant>(pattern_map[indices_label]);
        const auto depth_node = static_pointer_cast<op::Constant>(pattern_map[depth_label]);
        const auto on_node = static_pointer_cast<op::Constant>(pattern_map[on_label]);
        const auto off_node = static_pointer_cast<op::Constant>(pattern_map[off_label]);

        auto one_hot = static_pointer_cast<op::v1::OneHot>(m.get_match_root());
        const size_t axis = one_hot->get_axis();
        const auto output_shape = one_hot->get_output_shape(0);
        auto output_type = on_node->get_element_type();

        std::shared_ptr<op::Constant> replacement =
            fold_constant_one_hot<char>(indices_node, on_node, off_node, output_shape, axis);
        switch (output_type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in one_hot_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in one_hot_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in one_hot_callback");
            break;
        case element::Type_t::boolean:
            replacement =
                fold_constant_one_hot<char>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_one_hot<bfloat16>(
                indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::f16:
            replacement =
                fold_constant_one_hot<float16>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::f32:
            replacement =
                fold_constant_one_hot<float>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::f64:
            replacement =
                fold_constant_one_hot<double>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::i8:
            replacement =
                fold_constant_one_hot<int8_t>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::i16:
            replacement =
                fold_constant_one_hot<int16_t>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::i32:
            replacement =
                fold_constant_one_hot<int32_t>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::i64:
            replacement =
                fold_constant_one_hot<int64_t>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::u8:
            replacement =
                fold_constant_one_hot<uint8_t>(indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_one_hot<uint16_t>(
                indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_one_hot<uint32_t>(
                indices_node, on_node, off_node, output_shape, axis);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_one_hot<uint64_t>(
                indices_node, on_node, off_node, output_shape, axis);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };
    auto one_hot_matcher =
        make_shared<pattern::Matcher>(ont_hot_pattern, "ConstantFolding.ConstantOneHot");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(one_hot_matcher, one_hot_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
