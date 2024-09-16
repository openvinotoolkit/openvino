// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_scatter_elements_update12_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertScatterElementsUpdate12ToScatterElementsUpdate3::
    ConvertScatterElementsUpdate12ToScatterElementsUpdate3() {
    MATCHER_SCOPE(ConvertScatterElementsUpdate12ToScatterElementsUpdate3);

    const auto seu_v12_pattern = pattern::wrap_type<ov::op::v12::ScatterElementsUpdate>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto seu_v12 = ov::as_type_ptr<ov::op::v12::ScatterElementsUpdate>(m.get_match_root());
        if (!seu_v12 || transformation_callback(seu_v12) ||
            seu_v12->get_reduction() != ov::op::v12::ScatterElementsUpdate::Reduction::NONE) {
            return false;
        }

        const auto seu_v3 = std::make_shared<ov::op::v3::ScatterElementsUpdate>(seu_v12->input_value(0),
                                                                                seu_v12->input_value(1),
                                                                                seu_v12->input_value(2),
                                                                                seu_v12->input_value(3));

        seu_v3->set_friendly_name(seu_v12->get_friendly_name());
        copy_runtime_info(seu_v12, seu_v3);
        replace_node(seu_v12, seu_v3);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(seu_v12_pattern, matcher_name);
    register_matcher(m, callback);
}
