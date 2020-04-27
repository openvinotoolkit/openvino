// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_pad_to_pad_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertPadToPadIE::convert_pad() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto input_2 = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto input_3 = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto pad_1 = std::make_shared<ngraph::opset1::Pad>(input_0, input_1, input_2, op::PadMode::SYMMETRIC);
    auto pad_2 = std::make_shared<ngraph::opset1::Pad>(input_0, input_1, input_2, input_3, op::PadMode::CONSTANT);


    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto pad = std::dynamic_pointer_cast<ngraph::opset1::Pad> (m.get_match_root());
        if (!pad) {
            return false;
        }

        auto pad_ie = std::make_shared<ngraph::op::PadIE>(pad);
        if (pad_ie == nullptr)
            return false;
        pad_ie->set_friendly_name(pad->get_friendly_name());
        ngraph::copy_runtime_info(pad, pad_ie);
        ngraph::replace_node(pad, pad_ie);
        return true;
    };

    auto m1 = std::make_shared<ngraph::pattern::Matcher>(pad_1, "ConvertPadToPadIE");
    this->add_matcher(m1, callback, PassProperty::CHANGE_DYNAMIC_STATE);

    auto m2 = std::make_shared<ngraph::pattern::Matcher>(pad_2, "ConvertPadToPadIE");
    this->add_matcher(m2, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}