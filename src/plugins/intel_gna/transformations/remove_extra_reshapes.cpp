// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/remove_extra_reshapes.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

using namespace GNAPluginNS;

RemoveExtraReshapes::RemoveExtraReshapes() {
    MATCHER_SCOPE(RemoveExtraReshapes);
    const auto reshape = ngraph::pattern::wrap_type<ngraph::opset7::Reshape>(
        [](const ngraph::Output<ngraph::Node>& value) {
        return (value.get_node_shared_ptr()->get_input_shape(0) == value.get_node_shared_ptr()->get_output_shape(0));
    });
    const auto pooling = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({reshape});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        ngraph::replace_output_update_name(reshape_node->output(0), reshape_node->input_value(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pooling, matcher_name);
    this->register_matcher(m, callback);
}
