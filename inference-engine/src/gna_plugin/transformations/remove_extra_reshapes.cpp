// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/remove_extra_reshapes.hpp"

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(RemoveExtraReshapes, "RemoveExtraReshapes", 0);

RemoveExtraReshapes::RemoveExtraReshapes() {
    const auto reshape = ngraph::pattern::wrap_type<ngraph::opset7::Reshape>();
    const auto pooling = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({reshape});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_node = pattern_map.at(reshape).get_node_shared_ptr();
        if (reshape_node->get_input_shape(0) != reshape_node->get_output_shape(0)) {
            return false;
        }

        ngraph::replace_output_update_name(reshape_node->output(0), reshape_node->input_value(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(pooling, "RemoveExtraReshapes");
    this->register_matcher(m, callback);
}
