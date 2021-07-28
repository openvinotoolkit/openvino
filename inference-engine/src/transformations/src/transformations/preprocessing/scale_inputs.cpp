// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <ngraph/variant.hpp>
#include "ngraph/ops.hpp"
#include "ngraph/opsets/opset.hpp"
#include <ngraph/opsets/opset7.hpp>
#include "ngraph_ops/framework_node.hpp"
#include "pugixml.hpp"
#include "transformations/preprocessing/scale_inputs.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include <ngraph/pattern/op/or.hpp>

using namespace ngraph;
using namespace ngraph::pass;

NGRAPH_RTTI_DEFINITION(ngraph::pass::ScaleInputs, "ScaleInputs", 0);

using ConstantCreator = std::function<std::shared_ptr<opset7::Constant>(std::shared_ptr<Node>)>;

static bool matcher_callback(pattern::Matcher& m, ConstantCreator creator) {
    auto node = m.get_match_root();
    auto consumers = node->output(0).get_target_inputs();
    auto mul_const = creator(node);
    mul_const->set_friendly_name(node->get_friendly_name() + "/scale/Fused_Mul_Factor");
    auto new_op = std::make_shared<ngraph::opset7::Multiply>(node, mul_const);
    new_op->set_friendly_name(node->get_friendly_name() + "/scale/Fused_Mul");
    for (auto consumer : consumers) {
        consumer.replace_source_output(new_op);
    }
    return true;
}

void ScaleInputs::register_scale_matcher(ngraph::matcher_pass_callback callback) {
    auto param = pattern::wrap_type<op::Parameter>();
    auto m = std::make_shared<ngraph::pattern::Matcher>(param, "ScaleMatcher");
    register_matcher(m, callback);
}

ScaleInputs::ScaleInputs(float scale_factor): MatcherPass(), m_scale_factor(scale_factor) {
    register_scale_matcher(std::bind(matcher_callback, std::placeholders::_1, [&](std::shared_ptr<Node>) {
        return opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.f / m_scale_factor});
    }));
}

/// Calculate features dimension index based on input info
/// If initial_idx is not -1, use it if dimension equals to values_size
/// E.g. node_shape = {1,3,224,224}, scale_size=3 ==> Result will be "1" - dimension #1 is a 'features index'
static int guess_features_dim_idx(const std::shared_ptr<Node>& matched, size_t values_size, int initial_idx) {
    auto param_shape = matched->get_output_partial_shape(0);
    if (values_size == 1) {
        // Single scale value is always fine for any shape
        return 0;
    } else if (param_shape.rank().is_dynamic()) {
        throw ngraph_error("Scale of full dynamic input is not supported: " + matched->get_friendly_name());
    } else if (initial_idx >= 0) {
        if (initial_idx < param_shape.rank().get_length() &&
                param_shape[initial_idx].is_static() &&
                values_size == static_cast<size_t>(param_shape[initial_idx].get_length())) {
            return initial_idx;
        } else {
            throw ngraph_error("Feature dimension index " + std::to_string(initial_idx) +
                               " is invalid for node " + matched->get_friendly_name());
        }
    } else { // initial_idx is not specified
        auto rank_length = param_shape.rank().get_length();
        int found = 0;
        int foundIndex = -1;
        for (auto i = 0; i < rank_length; i++) {
            if (param_shape[i].is_static() &&
                    values_size == static_cast<size_t>(param_shape[i].get_length())) {
                found++;
                foundIndex = i;
            }
        }
        if (found == 1) {
            return foundIndex;
        } else {
            // Raise an exception, not clear how to calculate constant shape for inputs like {1,3,3,3}
            throw ngraph_error(
                    "Not clear how to apply scale vector to input " + matched->get_friendly_name());
        }
    }
}

ScaleInputs::ScaleInputs(const std::map<std::string, std::vector<float>>& scale_map, int features_dim_idx):
        MatcherPass(), m_scale_map(scale_map), m_features_dim_idx(features_dim_idx) {
    register_scale_matcher([this](pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto it = m_scale_map.find(node->get_friendly_name());
        if (it != m_scale_map.end()) {
            return matcher_callback(m, [&it, this](std::shared_ptr<Node> matched) {
                auto values = it->second;
                m_features_dim_idx = guess_features_dim_idx(matched, values.size(), m_features_dim_idx);
                auto param_shape = matched->get_output_partial_shape(0);
                std::vector<size_t> v(param_shape.rank().get_length(), 1);
                // Calculate shape of 'constant' based on node's partial shape and 'features dimension index'
                // E.g. node_shape = {1,3,224,224}, scale_size=3 ==> constant shape will be {1,3,1,1}
                ngraph::Shape constShape(v);
                constShape[m_features_dim_idx] = values.size();
                std::transform(values.begin(), values.end(), values.begin(), [](float val) -> float {
                    return 1.f / val;
                });
                return opset7::Constant::create(ngraph::element::f32, constShape, values);
            });
        }
        return false;
    });
}
