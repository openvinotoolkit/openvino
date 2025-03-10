// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_squeeze15_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertSqueeze15ToSqueeze0::ConvertSqueeze15ToSqueeze0() {
    MATCHER_SCOPE(ConvertSqueeze15ToSqueeze0);

    const auto& squeeze_v15_pattern = pattern::wrap_type<ov::op::v15::Squeeze>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& squeeze_v15 = ov::as_type_ptr<ov::op::v15::Squeeze>(m.get_match_root());
        if (!squeeze_v15 || transformation_callback(squeeze_v15)) {
            return false;
        }
        std::shared_ptr<op::v0::Squeeze> squeeze_v0;
        if (squeeze_v15->get_input_size() == 1) {
            squeeze_v0 = std::make_shared<op::v0::Squeeze>(squeeze_v15->input_value(0));
        } else if (squeeze_v15->get_input_size() == 2 && !squeeze_v15->get_allow_axis_skip()) {
            squeeze_v0 = std::make_shared<op::v0::Squeeze>(squeeze_v15->input_value(0), squeeze_v15->input_value(1));
        } else {
            return false;
        }
        squeeze_v0->set_friendly_name(squeeze_v15->get_friendly_name());
        copy_runtime_info(squeeze_v15, squeeze_v0);
        replace_node(squeeze_v15, squeeze_v0);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(squeeze_v15_pattern, matcher_name);
    register_matcher(m, callback);
}
