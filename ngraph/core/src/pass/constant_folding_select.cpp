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
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/reference/select.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_select(const shared_ptr<op::Constant>& selection,
                                              const shared_ptr<op::Constant>& t,
                                              const shared_ptr<op::Constant>& f,
                                              const shared_ptr<Node>& select)
{
    const Shape& out_shape = select->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();

    if (auto select_v0 = as_type_ptr<op::v0::Select>(select))
    {
        runtime::reference::select<T>(selection->get_data_ptr<char>(),
                                      t->get_data_ptr<T>(),
                                      f->get_data_ptr<T>(),
                                      data_ptr,
                                      shape_size(out_shape));
    }
    else if (auto select_v1 = as_type_ptr<op::v1::Select>(select))
    {
        runtime::reference::select<T>(selection->get_data_ptr<char>(),
                                      t->get_data_ptr<T>(),
                                      f->get_data_ptr<T>(),
                                      data_ptr,
                                      selection->get_shape(),
                                      t->get_shape(),
                                      f->get_shape(),
                                      select_v1->get_auto_broadcast());
    }

    return make_shared<op::Constant>(select->get_element_type(), out_shape, data_ptr);
}

void pass::ConstantFolding::construct_constant_select()
{
    auto selection_label = make_shared<pattern::op::Label>(
        element::boolean, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto t_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto f_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto select_v0_op = make_shared<op::v0::Select>(selection_label, t_label, f_label);
    auto select_v1_op = make_shared<op::v1::Select>(selection_label, t_label, f_label);

    auto constant_select_callback = [this, selection_label, t_label, f_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_select_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        const auto& selection_node =
            static_pointer_cast<op::Constant>(pattern_map[selection_label]);
        const auto& t_node = static_pointer_cast<op::Constant>(pattern_map[t_label]);
        const auto& f_node = static_pointer_cast<op::Constant>(pattern_map[f_label]);
        const auto& select = m.get_match_root();

        if (cf_is_disabled(select))
            return false;

        NGRAPH_CHECK(revalidate_and_ensure_static(select));

        std::shared_ptr<op::Constant> replacement;

        switch (select->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_select_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_select_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_select_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_select<char>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_select<bfloat16>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_select<float16>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_select<float>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_select<double>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_select<int8_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_select<int16_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_select<int32_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_select<int64_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_select<uint8_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_select<uint16_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_select<uint32_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_select<uint64_t>(selection_node, t_node, f_node, select);
            break;
        }

        replacement->set_friendly_name(m.get_match_root()->get_friendly_name());
        replace_node(m.get_match_root(), replacement);
        copy_runtime_info_to_target_inputs(m.get_match_root(), replacement);
        return true;
    };

    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(
        make_shared<pattern::Matcher>(select_v0_op, "ConstantFolding.ConstantSelectV0"),
        constant_select_callback,
        PassProperty::CHANGE_DYNAMIC_STATE);
    this->add_matcher(
        make_shared<pattern::Matcher>(select_v1_op, "ConstantFolding.ConstantSelectV1"),
        constant_select_callback,
        PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
