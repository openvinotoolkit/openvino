// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertAvgPool14ToAvgPool1::ConvertAvgPool14ToAvgPool1() {
    MATCHER_SCOPE(ConvertAvgPool14ToAvgPool1);

    const auto avg_pool_v14_pattern = pattern::wrap_type<ov::op::v14::AvgPool>();

    const matcher_pass_callback callback = [](pattern::Matcher& m) {
        const auto avg_pool_v14 = std::dynamic_pointer_cast<ov::op::v14::AvgPool>(m.get_match_root());
        if (!avg_pool_v14) {
            return false;
        }

        const auto rounding_type_v14 = avg_pool_v14->get_rounding_type();
        const auto rounding_type_v1 =
            rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH ? ov::op::RoundingType::CEIL : rounding_type_v14;

        const auto avg_pool_v1 = std::make_shared<ov::op::v1::AvgPool>(avg_pool_v14->input_value(0),
                                                                       avg_pool_v14->get_strides(),
                                                                       avg_pool_v14->get_pads_begin(),
                                                                       avg_pool_v14->get_pads_end(),
                                                                       avg_pool_v14->get_kernel(),
                                                                       avg_pool_v14->get_exclude_pad(),
                                                                       rounding_type_v1,
                                                                       avg_pool_v14->get_auto_pad());

        avg_pool_v1->set_friendly_name(avg_pool_v14->get_friendly_name());
        copy_runtime_info(avg_pool_v14, avg_pool_v1);
        replace_node(avg_pool_v14, avg_pool_v1);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(avg_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}
