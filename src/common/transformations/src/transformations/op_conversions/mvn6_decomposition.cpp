// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/mvn6_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::MVN6Decomposition::MVN6Decomposition() {
    MATCHER_SCOPE(MVN6Decomposition);
    // Decomposes MVN(x, axes) op if normalize_variance is false into sub-graph
    // x - ReduceMean(x, axes), if normalize_variance is true into sub-graph
    // (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    auto mvn = ov::pass::pattern::wrap_type<ov::op::v6::MVN>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto mvn_node = ov::as_type_ptr<ov::op::v6::MVN>(pattern_to_output.at(mvn).get_node_shared_ptr());

        if (mvn_node == nullptr || transformation_callback(mvn_node)) {
            return false;
        }

        const auto data = mvn_node->input_value(0);
        const auto axes = mvn_node->input_value(1);

        // (x - ReduceMean(x, axes))
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(data, axes, true);
        auto mean_normalization = std::make_shared<ov::op::v1::Subtract>(data, mean);

        if (!mvn_node->get_normalize_variance()) {
            mean_normalization->set_friendly_name(mvn_node->get_friendly_name());
            ov::copy_runtime_info(mvn_node, {mean, mean_normalization});
            ov::replace_node(mvn_node, mean_normalization);
        } else {
            // (x - ReduceMean(x, axes)) ^ 2
            auto sqr_const = ov::op::v0::Constant::create(data.get_element_type(), ov::Shape{1}, {2});
            auto sqr = std::make_shared<ov::op::v1::Power>(mean_normalization, sqr_const);
            // ReduceMean((x - ReduceMean(x, axes)) ^ 2)
            auto mean2 = std::make_shared<ov::op::v1::ReduceMean>(sqr, axes, true);

            auto eps = mvn_node->get_eps();
            auto eps_node = ov::op::v0::Constant::create(data.get_element_type(), ov::Shape{1}, {eps});
            auto eps_mode = mvn_node->get_eps_mode();

            std::shared_ptr<ov::op::v1::Add> eps_add;
            std::shared_ptr<ov::op::v0::Sqrt> sqrt;
            std::shared_ptr<ov::op::v1::Divide> div;

            if (eps_mode == op::MVNEpsMode::INSIDE_SQRT) {
                // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
                eps_add = std::make_shared<ov::op::v1::Add>(mean2, eps_node);
                sqrt = std::make_shared<ov::op::v0::Sqrt>(eps_add);
                // (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
                div = std::make_shared<ov::op::v1::Divide>(mean_normalization, sqrt);
            } else if (eps_mode == op::MVNEpsMode::OUTSIDE_SQRT) {
                // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
                sqrt = std::make_shared<ov::op::v0::Sqrt>(mean2);
                eps_add = std::make_shared<ov::op::v1::Add>(sqrt, eps_node);
                // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
                div = std::make_shared<ov::op::v1::Divide>(mean_normalization, eps_add);
            } else {
                return false;
            }

            div->set_friendly_name(mvn_node->get_friendly_name());
            ov::copy_runtime_info(mvn_node, {mean, mean_normalization, sqr, mean2, eps_node, eps_add, sqrt, div});
            ov::replace_node(mvn_node, div);
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mvn, matcher_name);
    register_matcher(m, callback);
}
