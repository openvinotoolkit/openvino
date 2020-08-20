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
#include "ngraph/op/dequantize.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

template <class QUANT, class REAL>
shared_ptr<op::Constant> fold_constant_dequantize(shared_ptr<op::Constant> constant,
                                                  shared_ptr<op::Dequantize> dequant,
                                                  shared_ptr<op::Constant> scale,
                                                  shared_ptr<op::Constant> offset)
{
    const Shape& out_shape = constant->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(REAL));
    REAL* data_ptr = buffer.get_ptr<REAL>();

    runtime::reference::dequantize<QUANT, REAL>(constant->get_data_ptr<QUANT>(),
                                                scale->get_data_ptr<REAL>(),
                                                offset->get_data_ptr<QUANT>(),
                                                data_ptr,
                                                constant->get_shape(),
                                                scale->get_shape(),
                                                dequant->get_axes());

    return make_shared<op::Constant>(dequant->get_element_type(), out_shape, data_ptr);
}

void pass::ConstantFolding::construct_constant_dequantize()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::u8, Shape{2}, pattern::has_class<op::Constant>());
    auto dq_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto dq_offset = op::Constant::create(element::u8, Shape{}, {1});
    auto dequant_op =
        make_shared<op::Dequantize>(constant_label, dq_scale, dq_offset, element::f32, AxisSet{});
    auto dequant = make_shared<pattern::op::Label>(dequant_op, nullptr, NodeVector{dequant_op});

    auto constant_dequantize_callback = [this, constant_label, dequant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dequantize_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = as_type_ptr<op::Constant>(pattern_map[constant_label]);
        auto dequant_match = pattern_map[dequant];
        auto dequantize_op = as_type_ptr<op::Dequantize>(dequant_match);

        if (cf_is_disabled(dequantize_op))
            return false;

        auto scale = as_type_ptr<op::Constant>(dequant_match->input_value(1).get_node_shared_ptr());
        auto offset =
            as_type_ptr<op::Constant>(dequant_match->input_value(2).get_node_shared_ptr());

        NGRAPH_CHECK(revalidate_and_ensure_static(dequantize_op));
        auto type = constant_match->get_element_type();

        if (dequant_match->get_element_type() != element::f32)
        {
            return false;
        }

        if (type == element::u8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_dequantize<uint8_t, float>(
                             constant_match, dequantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(m.get_match_root(),
                         fold_constant_dequantize<int8_t, float>(
                             constant_match, dequantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto dequantize_matcher =
        make_shared<pattern::Matcher>(dequant, "ConstantFolding.ConstantDequantize");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(
        dequantize_matcher, constant_dequantize_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
