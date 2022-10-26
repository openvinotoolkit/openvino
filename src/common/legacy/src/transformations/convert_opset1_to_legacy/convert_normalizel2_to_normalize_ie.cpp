// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_normalizel2_to_normalize_ie.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include "legacy/ngraph_ops/normalize_ie.hpp"

ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE::ConvertNormalizeL2WithMulToNormalizeIE() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto axis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});

    auto normalize = std::make_shared<ngraph::opset1::NormalizeL2>(input_0, axis, 0.0f, ngraph::op::EpsMode::ADD);
    auto mul = std::make_shared<ngraph::opset1::Multiply> (normalize, input_1);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto mul = std::dynamic_pointer_cast<ngraph::opset1::Multiply> (m.get_match_root());
        if (!mul) return false;

        auto normalize = std::dynamic_pointer_cast<ngraph::opset1::NormalizeL2> (mul->input(0).get_source_output().get_node_shared_ptr());
        auto weights_output = mul->input(1).get_source_output();
        if (!normalize) {
            normalize = std::dynamic_pointer_cast<ngraph::opset1::NormalizeL2> (mul->input(1).get_source_output().get_node_shared_ptr());
            weights_output = mul->input(1).get_source_output();
            if (!normalize) return false;
        }

        auto const_axis = std::dynamic_pointer_cast<ngraph::opset1::Constant> (normalize->input(1).get_source_output().get_node_shared_ptr());
        if (!const_axis) return false;

        //  Handle two cases:
        //  1. When Mul has weights input as Broadcast
        //  2. When Mul has weights as Constant

        auto broadcast = std::dynamic_pointer_cast<ngraph::opset1::Broadcast> (weights_output.get_node_shared_ptr());
        auto constant = std::dynamic_pointer_cast<ngraph::opset1::Constant> (weights_output.get_node_shared_ptr());

        if (broadcast) {
            constant = std::dynamic_pointer_cast<ngraph::opset1::Constant> (broadcast->input(0).get_source_output().get_node_shared_ptr());
        }

        if (!constant) {
            return false;
        }

        //  Replace NormalizeL2 with NormalizeIE operation

        auto axis = const_axis->cast_vector<size_t>();
        bool across_spatial = !(axis.size() == 1 && axis[0] == 1);
        bool channel_shared = (constant->get_shape().size() == 1);

        auto normalize_ie = std::make_shared<ngraph::op::NormalizeIE> (normalize->input(0).get_source_output(),
                                                                       constant->output(0),
                                                                       normalize->get_eps(),
                                                                       across_spatial,
                                                                       channel_shared,
                                                                       normalize->get_element_type());

        normalize_ie->set_friendly_name(mul->get_friendly_name());
        ngraph::copy_runtime_info({normalize, mul}, normalize_ie);
        ngraph::replace_node(mul, normalize_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "CPUFusion.ConvertNormalizeL2WithMulToNormalizeIE");
    this->register_matcher(m, callback);
}

ngraph::pass::ConvertNormalizeL2ToLegacyMatcher::ConvertNormalizeL2ToLegacyMatcher() {
    auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto axis = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto normalize = std::make_shared<ngraph::opset1::NormalizeL2>(input_0, axis, 0.0f, ngraph::op::EpsMode::ADD);

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto normalize = std::dynamic_pointer_cast<ngraph::opset1::NormalizeL2> (m.get_match_root());
        if (!normalize) return false;

        auto const_axis = std::dynamic_pointer_cast<ngraph::opset1::Constant> (normalize->input(1).get_source_output().get_node_shared_ptr());
        if (!const_axis) return false;

        //  Replace NormalizeL2 with NormalizeIE operation
        auto axis = const_axis->cast_vector<size_t>();
        bool across_channels = !(axis.size() == 1 && axis[0] == 1);
        bool channel_shared = true;

        auto scale = std::make_shared<ngraph::opset1::Constant>(normalize->output(0).get_element_type(), Shape{1}, std::vector<float>{1.0});

        auto normalize_ie = std::make_shared<ngraph::op::NormalizeIE> (normalize->input(0).get_source_output(),
                                                                       scale->output(0),
                                                                       normalize->get_eps(),
                                                                       across_channels,
                                                                       channel_shared,
                                                                       normalize->get_element_type());

        normalize_ie->set_friendly_name(normalize->get_friendly_name());
        ngraph::copy_runtime_info(normalize, normalize_ie);
        ngraph::replace_node(normalize, normalize_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(normalize, "ConvertNormalizeL2ToNormalizeIE");
    this->register_matcher(m, callback);
}
