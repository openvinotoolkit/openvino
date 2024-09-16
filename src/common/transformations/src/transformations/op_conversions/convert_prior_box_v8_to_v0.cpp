// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/prior_box.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertPriorBox8To0::ConvertPriorBox8To0() {
    MATCHER_SCOPE(ConvertPriorBox8To0);

    auto prior_box_v8 = pattern::wrap_type<ov::op::v8::PriorBox>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto prior_box_v8_node = ov::as_type_ptr<ov::op::v8::PriorBox>(m.get_match_root());
        if (!prior_box_v8_node)
            return false;

        ov::op::v8::PriorBox::Attributes attrs_v8 = prior_box_v8_node->get_attrs();
        if (!attrs_v8.min_max_aspect_ratios_order)
            return false;

        ov::op::v0::PriorBox::Attributes attrs_v0;
        attrs_v0.min_size = attrs_v8.min_size;
        attrs_v0.max_size = attrs_v8.max_size;
        attrs_v0.aspect_ratio = attrs_v8.aspect_ratio;
        attrs_v0.density = attrs_v8.density;
        attrs_v0.fixed_ratio = attrs_v8.fixed_ratio;
        attrs_v0.fixed_size = attrs_v8.fixed_size;
        attrs_v0.clip = attrs_v8.clip;
        attrs_v0.flip = attrs_v8.flip;
        attrs_v0.step = attrs_v8.step;
        attrs_v0.offset = attrs_v8.offset;
        attrs_v0.variance = attrs_v8.variance;
        attrs_v0.scale_all_sizes = attrs_v8.scale_all_sizes;

        auto prior_box_v0 = std::make_shared<ov::op::v0::PriorBox>(prior_box_v8_node->input_value(0),
                                                                   prior_box_v8_node->input_value(1),
                                                                   attrs_v0);
        prior_box_v0->set_friendly_name(prior_box_v8_node->get_friendly_name());
        ov::copy_runtime_info(prior_box_v8_node, prior_box_v0);
        ov::replace_node(prior_box_v8_node, prior_box_v0);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prior_box_v8, matcher_name);
    register_matcher(m, callback);
}
