// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_pad12_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertPad12ToPad1::ConvertPad12ToPad1() {
    MATCHER_SCOPE(ConvertPad12ToPad1);

    const auto pad_v12_pattern = pattern::wrap_type<ov::op::v12::Pad>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto pad_v12 = ov::as_type_ptr<ov::op::v12::Pad>(m.get_match_root());
        if (!pad_v12 || transformation_callback(pad_v12)) {
            return false;
        }

        std::shared_ptr<ov::Node> pad_v1;
        if (pad_v12->get_input_size() == 4) {
            pad_v1 = std::make_shared<ov::op::v1::Pad>(pad_v12->input_value(0),
                                                       pad_v12->input_value(1),
                                                       pad_v12->input_value(2),
                                                       pad_v12->input_value(3),
                                                       pad_v12->get_pad_mode());
        } else {
            const auto pad_value =
                ov::op::v0::Constant::create(pad_v12->input_value(0).get_element_type(), ov::Shape{}, {0});

            pad_v1 = std::make_shared<ov::op::v1::Pad>(pad_v12->input_value(0),
                                                       pad_v12->input_value(1),
                                                       pad_v12->input_value(2),
                                                       pad_value,
                                                       pad_v12->get_pad_mode());
        }
        pad_v1->set_friendly_name(pad_v12->get_friendly_name());
        copy_runtime_info(pad_v12, pad_v1);
        replace_node(pad_v12, pad_v1);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(pad_v12_pattern, matcher_name);
    register_matcher(m, callback);
}
