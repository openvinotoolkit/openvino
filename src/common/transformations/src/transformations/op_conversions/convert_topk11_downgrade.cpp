// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_topk11_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertTopK11ToTopK3::ConvertTopK11ToTopK3() {
    MATCHER_SCOPE(ConvertTopK11ToTopK3);

    const auto topk_v11_pattern = pattern::wrap_type<ov::op::v11::TopK>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto topk_v11 = ov::as_type_ptr<ov::op::v11::TopK>(m.get_match_root());
        if (!topk_v11 || transformation_callback(topk_v11)) {
            return false;
        }

        // downgrade even if stable attribute is True
        // this is needed to provide backward-compatibility
        // and operation working in the plugins that have not yet added stable mode

        const auto topk_v3 = std::make_shared<ov::op::v3::TopK>(topk_v11->input_value(0),
                                                                topk_v11->input_value(1),
                                                                topk_v11->get_axis(),
                                                                topk_v11->get_mode(),
                                                                topk_v11->get_sort_type(),
                                                                topk_v11->get_index_element_type());

        topk_v3->set_friendly_name(topk_v11->get_friendly_name());
        copy_runtime_info(topk_v11, topk_v3);
        replace_node(topk_v11, topk_v3);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(topk_v11_pattern, matcher_name);
    register_matcher(m, callback);
}
