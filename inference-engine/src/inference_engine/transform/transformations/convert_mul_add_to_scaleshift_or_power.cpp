// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include <ngraph_ops/scaleshift.hpp>
#include <ngraph_ops/power.hpp>

#include "convert_mul_add_to_scaleshift_or_power.hpp"

#include "ngraph/op/add.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/pattern/matcher.hpp"

#include "ngraph/graph_util.hpp"

CONVERSION_RESULT check_constant(const std::shared_ptr<ngraph::op::Constant> & constant,
                                 const std::vector<int64_t> & output_shape) {
    if (!constant) return CONVERSION_RESULT::NONE;

    auto input_shape = constant->get_shape();

    // This is feature dimension index from right side (ex. for NCDHW it's equal to 3).
    const size_t feature_index = output_shape.size() - 2;
    if (input_shape.size() < feature_index) return CONVERSION_RESULT::NONE;

    bool is_power = false;
    auto in_it = input_shape.rbegin();
    auto out_it = output_shape.rbegin();
    for (int idx = 0; in_it != input_shape.rend() && out_it != output_shape.rend(); ++in_it, ++out_it, ++idx) {
        if (idx != feature_index && *in_it != 1) {
            return CONVERSION_RESULT::NONE;
        }

        if (idx == feature_index && *in_it == 1) {
            is_power = true;
        } else if (idx == feature_index && *in_it != *out_it) {
            return CONVERSION_RESULT::NONE;
        }
    }

    return is_power ? CONVERSION_RESULT::POWER : CONVERSION_RESULT::SCALE_SHIFT;
}

std::shared_ptr<ngraph::Node> normalize_constant(const std::shared_ptr<ngraph::op::Constant> & constant,
                                                 const ngraph::Shape & shape) {
    auto const_shape = constant->get_shape();
    if (const_shape.size() == shape.size()) {
        return std::dynamic_pointer_cast<ngraph::Node> (constant);
    }
    int cnt = shape.size() - const_shape.size();
    for (int i = 0; i < cnt; ++i) {
        const_shape.insert(const_shape.begin(), 1);
    }

    auto order = ngraph::get_default_order(constant->get_shape());
    auto reshape = std::make_shared<ngraph::op::Reshape>(constant, order, const_shape);
    return std::dynamic_pointer_cast<ngraph::Node> (reshape);
}

CONVERSION_RESULT check_dyn_broadcast(const std::shared_ptr<ngraph::op::DynBroadcast> & broadcast) {
    if (!broadcast) return CONVERSION_RESULT::NONE;

    auto shape_node = std::dynamic_pointer_cast<ngraph::op::Constant>(broadcast->get_inputs()[1].get_output().get_node());
    auto output_shape = shape_node->get_vector<int64_t>();

    auto data_node =  std::dynamic_pointer_cast<ngraph::op::Constant>(broadcast->get_inputs()[0].get_output().get_node());
    if (!data_node) return CONVERSION_RESULT::NONE;

    return check_constant(data_node, output_shape);
}

void ngraph::pass::ConvertMulAddToScaleShiftOrPower::convert_mul_add_to_scaleshift_or_power() {
    Shape shape{2, 2, 1, 1};
    auto data_batch = std::make_shared<pattern::op::Label>(element::f32, shape);

    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto shp1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto axs1 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto broadcast1 = std::make_shared<ngraph::op::DynBroadcast>(weights, shp1, axs1);

    auto bias = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto shp2 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto axs2 = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto broadcast2 = std::make_shared<ngraph::op::DynBroadcast>(bias, shp2, axs2);

    auto mul = std::make_shared<ngraph::op::Multiply>(data_batch, broadcast1);
    auto add = std::make_shared<ngraph::op::Add>(mul, broadcast2);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto add_node = std::dynamic_pointer_cast<ngraph::op::Add> (m.get_match_root());
        if (!add_node) {
            return false;
        }

        auto mul_node = std::dynamic_pointer_cast<ngraph::op::Multiply> (add_node->get_argument(0));
        auto broadcast_bias_node = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (add_node->get_argument(1));
        if (!mul_node) {
            mul_node = std::dynamic_pointer_cast<ngraph::op::Multiply> (add_node->get_argument(1));
            broadcast_bias_node = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (add_node->get_argument(0));
        }

        auto data_node = std::dynamic_pointer_cast<Node> (mul_node->get_argument(0));
        auto broadcast_weights_node = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (mul_node->get_argument(1));
        if (!broadcast_weights_node) {
            data_node = std::dynamic_pointer_cast<Node> (mul_node->get_argument(1));
            broadcast_weights_node = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (mul_node->get_argument(0));
        }

        // Check that DynBroadcast inputs are applicable for ScaleShift
        auto res1 = check_dyn_broadcast(broadcast_weights_node);
        auto res2 = check_dyn_broadcast(broadcast_bias_node);

        // TODO: in case Power|Scaleshift results, do we need to expand weights|biases to ScaleShift?
        if ((res1 == CONVERSION_RESULT::NONE || res2 == CONVERSION_RESULT::NONE) || (res1 != res2)) {
            return false;
        }

        auto weights_node = std::dynamic_pointer_cast<ngraph::op::Constant>(broadcast_weights_node->get_inputs()[0].get_output().get_node());
        auto bias_node = std::dynamic_pointer_cast<ngraph::op::Constant>(broadcast_bias_node->get_inputs()[0].get_output().get_node());

        if (res1 == CONVERSION_RESULT::SCALE_SHIFT) {
            auto scaleshift = std::make_shared<ngraph::op::ScaleShiftIE>(data_node,
                                                                         normalize_constant(weights_node, add_node->get_shape()),
                                                                         normalize_constant(bias_node, add_node->get_shape()));
            scaleshift->set_friendly_name(add_node->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<Node>(scaleshift));
        } else {
            // TODO: currently only FP32 support
            if (weights_node->get_element_type() != element::f32) return false;
            if (bias_node->get_element_type() != element::f32) return false;

            auto power = std::make_shared<ngraph::op::PowerIE>(data_node,
                                                             1.,
                                                             *weights_node->get_vector<float>().begin(),
                                                             *bias_node->get_vector<float>().begin());
            power->set_friendly_name(add_node->get_friendly_name());
            ngraph::replace_node(m.get_match_root(), std::dynamic_pointer_cast<Node>(power));
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "CPUFusion.MulAddToScaleShiftOrPower");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
