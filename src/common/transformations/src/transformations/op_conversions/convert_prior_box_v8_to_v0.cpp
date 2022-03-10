// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_prior_box_v8_to_v0.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ngraph::pass::ConvertPriorBox8To0::ConvertPriorBox8To0() {
    MATCHER_SCOPE(ConvertPriorBox8To0);

    auto prior_box_v8 = pattern::wrap_type<ngraph::opset8::PriorBox>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto prior_box_v8_node = std::dynamic_pointer_cast<ngraph::opset8::PriorBox>(m.get_match_root());
        if (!prior_box_v8_node)
            return false;

        ngraph::opset8::PriorBox::Attributes attrs_v8 = prior_box_v8_node->get_attrs();
        if (!attrs_v8.min_max_aspect_ratios_order)
            return false;

        ngraph::opset1::PriorBox::Attributes attrs_v0;
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

        auto prior_box_v0 = std::make_shared<ngraph::opset1::PriorBox>(prior_box_v8_node->input_value(0),
                                                                       prior_box_v8_node->input_value(1),
                                                                       attrs_v0);
        prior_box_v0->set_friendly_name(prior_box_v8_node->get_friendly_name());
        ngraph::copy_runtime_info(prior_box_v8_node, prior_box_v0);
        ngraph::replace_node(prior_box_v8_node, prior_box_v0);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(prior_box_v8, matcher_name);
    register_matcher(m, callback);
}
