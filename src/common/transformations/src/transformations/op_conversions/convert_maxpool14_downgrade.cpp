// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/op_conversions/convert_maxpool14_to_maxpool8_downgrade.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertMaxPool14ToMaxPool8::ConvertMaxPool14ToMaxPool8() {
    MATCHER_SCOPE(ConvertMaxPool14ToMaxPool8);

    const auto max_pool_v14_pattern = pattern::wrap_type<ov::op::v14::MaxPool>();

    const matcher_pass_callback callback = [this](pattern::Matcher& m) {
        const auto max_pool_v14 = std::dynamic_pointer_cast<ov::op::v14::MaxPool>(m.get_match_root());
        if (!max_pool_v14) {
            return false;
        }

        const auto rounding_type_v14 = max_pool_v14->get_rounding_type();
        const auto rounding_type_v8 =
            rounding_type_v14 == ov::op::RoundingType::CEIL_TORCH ? ov::op::RoundingType::CEIL : rounding_type_v14;

        const auto max_pool_v8 = std::make_shared<ov::op::v8::MaxPool>(max_pool_v14->input_value(0),
                                                                       max_pool_v14->get_strides(),
                                                                       max_pool_v14->get_dilations(),
                                                                       max_pool_v14->get_pads_begin(),
                                                                       max_pool_v14->get_pads_end(),
                                                                       max_pool_v14->get_kernel(),
                                                                       rounding_type_v8,
                                                                       max_pool_v14->get_auto_pad(),
                                                                       max_pool_v14->get_index_element_type(),
                                                                       max_pool_v14->get_axis());

        max_pool_v8->set_friendly_name(max_pool_v14->get_friendly_name());
        copy_runtime_info(max_pool_v14, max_pool_v8);
        replace_node(max_pool_v14, max_pool_v8);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(max_pool_v14_pattern, matcher_name);
    register_matcher(m, callback);
}
