// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_prelu.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::ReshapePRelu, "ReshapePRelu", 0);

MKLDNNPlugin::ReshapePRelu::ReshapePRelu() {
    auto prelu = ngraph::pattern::wrap_type<ngraph::opset1::PRelu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                                    ngraph::pattern::any_input(ngraph::pattern::has_static_shape())});

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(m.get_match_root());
        if (!prelu || ngraph::shape_size(prelu->get_input_shape(1)) == 1 || prelu->get_input_shape(1).size() != 1) {
            return false;
        }
        const auto prelu_shape = prelu->input_value(0).get_shape();
        const auto slope_shape = prelu->input_value(1).get_shape();
        ngraph::Shape new_shape(prelu_shape.size(), 1);
        const auto slope_dim = slope_shape[0];
        const auto channel_dim_idx = prelu_shape.size() > 1 ? 1 : 0;
        if (slope_dim != prelu_shape[channel_dim_idx]) {
            return false;
        }
        new_shape[channel_dim_idx] = slope_dim;

        auto slope = ngraph::op::util::reshapeTo(prelu->input_value(1), new_shape);
        auto new_prelu = std::make_shared<ngraph::opset1::PRelu>(prelu->input(0).get_source_output(), slope);
        new_prelu->set_friendly_name(prelu->get_friendly_name());
        ngraph::copy_runtime_info(prelu, new_prelu);
        ngraph::replace_node(prelu, new_prelu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "ReshapePRelu");
    this->register_matcher(m, callback);
}
