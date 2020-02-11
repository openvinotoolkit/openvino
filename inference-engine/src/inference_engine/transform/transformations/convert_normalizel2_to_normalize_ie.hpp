// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/op/fused/normalize_l2.hpp"
#include "ngraph/op/constant.hpp"

#include "ngraph_ops/normalize_ie.hpp"


namespace ngraph {
namespace pass {

class ConvertNormalizeL2WithMulToNormalizeIE;
class ConvertNormalizeL2ToNormalizeIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNormalizeL2WithMulToNormalizeIE: public ngraph::pass::GraphRewrite {
public:
    ConvertNormalizeL2WithMulToNormalizeIE() : GraphRewrite() {
        convert_normalize_l2_with_mul();
    }

private:
    void convert_normalize_l2_with_mul() {
        auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto input_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto axis = std::make_shared<ngraph::op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});

        auto normalize = std::make_shared<ngraph::op::NormalizeL2>(input_0, axis, 0, ngraph::op::EpsMode::ADD);
        auto mul = std::make_shared<ngraph::op::v1::Multiply> (normalize, input_1);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto mul = std::dynamic_pointer_cast<ngraph::op::v1::Multiply> (m.get_match_root());
            if (!mul) return false;

            auto normalize = std::dynamic_pointer_cast<ngraph::op::NormalizeL2> (mul->input(0).get_source_output().get_node_shared_ptr());
            auto weights_output = mul->input(1).get_source_output();
            if (!normalize) {
                normalize = std::dynamic_pointer_cast<ngraph::op::NormalizeL2> (mul->input(1).get_source_output().get_node_shared_ptr());
                weights_output = mul->input(1).get_source_output();
                if (!normalize) return false;
            }

            auto const_axis = std::dynamic_pointer_cast<ngraph::op::Constant> (normalize->input(1).get_source_output().get_node_shared_ptr());
            if (!const_axis) return false;

            //  Handle two cases:
            //  1. When Mul has weights input as Broadcast
            //  2. When Mul has weights as Constant

            auto broadcast = std::dynamic_pointer_cast<ngraph::op::v1::Broadcast> (weights_output.get_node_shared_ptr());
            auto constant = std::dynamic_pointer_cast<ngraph::op::Constant> (weights_output.get_node_shared_ptr());

            if (broadcast) {
                constant = std::dynamic_pointer_cast<ngraph::op::Constant> (broadcast->input(0).get_source_output().get_node_shared_ptr());
            }

            if (!constant) {
                return false;
            }

            //  Replace NormalizeL2 with NormalizeIE operation

            auto axis = const_axis->get_vector<size_t>();
            bool across_spatial = !(axis.size() == 1 && axis[0] == 1);
            bool channel_shared = (constant->get_shape().size() == 1);

            auto normalize_ie = std::make_shared<ngraph::op::NormalizeIE> (normalize->input(0).get_source_output(),
                                                                           constant->output(0),
                                                                           normalize->get_eps(),
                                                                           across_spatial,
                                                                           channel_shared);

            normalize_ie->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), normalize_ie);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(mul, "CPUFusion.ConvertNormalizeL2WithMulToNormalizeIE");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};

class ngraph::pass::ConvertNormalizeL2ToNormalizeIE: public ngraph::pass::GraphRewrite {
public:
    ConvertNormalizeL2ToNormalizeIE() : GraphRewrite() {
        convert_normalize_l2();
    }

private:
    void convert_normalize_l2() {
        auto input_0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto axis = std::make_shared<ngraph::op::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
        auto normalize = std::make_shared<ngraph::op::NormalizeL2>(input_0, axis, 0, ngraph::op::EpsMode::ADD);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto normalize = std::dynamic_pointer_cast<ngraph::op::NormalizeL2> (m.get_match_root());
            if (!normalize) return false;

            auto const_axis = std::dynamic_pointer_cast<ngraph::op::Constant> (normalize->input(1).get_source_output().get_node_shared_ptr());
            if (!const_axis) return false;

            //  Replace NormalizeL2 with NormalizeIE operation
            auto axis = const_axis->get_vector<size_t>();
            bool across_channels = !(axis.size() == 1 && axis[0] == 1);
            bool channel_shared = true;

            auto scale = std::make_shared<ngraph::op::Constant>(normalize->get_input_element_type(0), Shape{1}, std::vector<float>{1.0});

            auto normalize_ie = std::make_shared<ngraph::op::NormalizeIE> (normalize->input(0).get_source_output(),
                                                                           scale->output(0),
                                                                           normalize->get_eps(),
                                                                           across_channels,
                                                                           channel_shared);

            normalize_ie->set_friendly_name(m.get_match_root()->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), normalize_ie);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(normalize, "CPUFusion.ConvertNormalizeL2ToNormalizeIE");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
