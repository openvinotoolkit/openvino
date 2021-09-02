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
    auto input = ngraph::pattern::any_input(ngraph::pattern::has_static_rank());
    auto slope_constant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto prelu = ngraph::pattern::wrap_type<ngraph::opset1::PRelu>({ input, slope_constant });

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto prelu = std::dynamic_pointer_cast<ngraph::opset1::PRelu>(m.get_match_root());

        const auto slope_shape = prelu->input_value(1).get_shape();
        const auto prelu_pshape = prelu->input_value(0).get_partial_shape();
        const auto prelu_rank = prelu_pshape.rank();
        if (!prelu || prelu_rank.is_dynamic() || prelu_rank.get_length() == 1 || ngraph::shape_size(slope_shape) == 1 || slope_shape.size() != 1) {
            return false;
        }

        const auto slope_dim = slope_shape[0];
        const auto channel_dim_idx = 1;
        if (!prelu_pshape[channel_dim_idx].is_dynamic() && slope_dim != prelu_pshape[channel_dim_idx].get_length()) {
            return false;
        }

        ngraph::Shape new_slope_shape(prelu_rank.get_length(), 1);
        new_slope_shape[channel_dim_idx] = slope_dim;

        auto reshape_constant = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ new_slope_shape.size() }, new_slope_shape);
        auto new_slope = ngraph::op::util::make_try_fold<ngraph::opset1::Reshape>(prelu->input_value(1), reshape_constant, true);
        auto new_prelu = prelu->clone_with_new_inputs({ prelu->input_value(0), new_slope });
        new_prelu->set_friendly_name(prelu->get_friendly_name());

        ngraph::copy_runtime_info(prelu, new_prelu);
        ngraph::replace_node(prelu, new_prelu);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(prelu, "ReshapePRelu");
    this->register_matcher(m, callback);
}
