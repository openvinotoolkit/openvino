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

#include <numeric>

#include "constant_folding.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <typename T, typename R>
shared_ptr<op::Constant> fold_constant_dyn_reshape(shared_ptr<op::Constant> constant_data,
                                                   R dyn_reshape)
{
    // v1::Reshape and v0::DynReshape do not allow data transposes.
    return make_shared<op::Constant>(dyn_reshape->get_element_type(),
                                     dyn_reshape->get_shape(),
                                     constant_data->get_data_ptr<T>());
}

template <typename R>
std::shared_ptr<Node> do_fold(R dyn_reshape_match, shared_ptr<op::Constant> constant_data_match)
{
    std::shared_ptr<Node> replacement;
    auto type = dyn_reshape_match->get_element_type();
    switch (type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false,
                     "Encountered 'undefined' element type in constant_dyn_reshape_callback");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_dyn_reshape_callback");
        break;
    case element::Type_t::u1:
        NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_dyn_reshape_callback");
        break;
    case element::Type_t::boolean:
        replacement = fold_constant_dyn_reshape<char>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::bf16:
        replacement = fold_constant_dyn_reshape<bfloat16>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::f16:
        replacement = fold_constant_dyn_reshape<float16>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::f32:
        replacement = fold_constant_dyn_reshape<float>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::f64:
        replacement = fold_constant_dyn_reshape<double>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::i8:
        replacement = fold_constant_dyn_reshape<int8_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::i16:
        replacement = fold_constant_dyn_reshape<int16_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::i32:
        replacement = fold_constant_dyn_reshape<int32_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::i64:
        replacement = fold_constant_dyn_reshape<int64_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::u8:
        replacement = fold_constant_dyn_reshape<uint8_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::u16:
        replacement = fold_constant_dyn_reshape<uint16_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::u32:
        replacement = fold_constant_dyn_reshape<uint32_t>(constant_data_match, dyn_reshape_match);
        break;
    case element::Type_t::u64:
        replacement = fold_constant_dyn_reshape<uint64_t>(constant_data_match, dyn_reshape_match);
        break;
    }
    return replacement;
}

void pass::ConstantFolding::construct_constant_dyn_reshape()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
    auto constant_shape_label =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());
    auto reshape_v1 =
        make_shared<op::v1::Reshape>(constant_data_label, constant_shape_label, false);

    // Note: No need to capture or consider constant_shape_label, because
    // shape propagation will have transferred the info to dyn_reshape's
    // output.
    auto constant_reshape_v1_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_reshape_v1_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_data_match =
            static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto match_root = m.get_match_root();
        NGRAPH_CHECK(revalidate_and_ensure_static(match_root));
        shared_ptr<Node> replacement;
        replacement =
            do_fold(static_pointer_cast<op::v1::Reshape>(match_root), constant_data_match);
        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto reshape_v1_matcher =
        make_shared<pattern::Matcher>(reshape_v1, "ConstantFolding.ConstantReshapev1");
    this->add_matcher(
        reshape_v1_matcher, constant_reshape_v1_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
