// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_scatter_nd_update15_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertScatterNDUpdate15ToScatterNDUpdate3::ConvertScatterNDUpdate15ToScatterNDUpdate3() {
    MATCHER_SCOPE(ConvertScatterNDUpdate15ToScatterNDUpdate3);

    const auto scatter_v15_pattern = pattern::wrap_type<ov::op::v15::ScatterNDUpdate>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto scatter_v15 = ov::as_type_ptr<ov::op::v15::ScatterNDUpdate>(m.get_match_root());
        if (!scatter_v15 || transformation_callback(scatter_v15)) {
            return false;
        }
        if (scatter_v15->get_reduction() != ov::op::v15::ScatterNDUpdate::Reduction::NONE) {
            return false;
        }
        const auto scatter_v3 = std::make_shared<ov::op::v3::ScatterNDUpdate>(scatter_v15->input_value(0),
                                                                              scatter_v15->input_value(1),
                                                                              scatter_v15->input_value(2));

        scatter_v3->set_friendly_name(scatter_v15->get_friendly_name());
        copy_runtime_info(scatter_v15, scatter_v3);
        replace_node(scatter_v15, scatter_v3);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(scatter_v15_pattern, matcher_name);
    register_matcher(m, callback);
}
