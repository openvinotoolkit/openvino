// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mvn_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::Matcher;
using ov::pass::pattern::wrap_type;
using ov::pass::pattern::op::Or;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v6 = ov::op::v6;
namespace op_util = ov::op::util;
template <class T>
std::function<bool(ov::Output<ov::Node>)> value_is_equal_to(const std::vector<T>& ref_values) {
    return [ref_values](ov::Output<ov::Node> output) -> bool {
        auto node = output.get_node_shared_ptr();
        if (auto const_node = ov::as_type_ptr<v0::Constant>(node)) {
            return const_node->template cast_vector<T>() == ref_values;
        }
        return false;
    };
}

ov::pass::MVNFusionWithoutConstants::MVNFusionWithoutConstants() {
    MATCHER_SCOPE(MVNFusionWithoutConstants);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps)
    auto x = any_input();

    // (x - ReduceMean(x, axes))
    //     `------mean1-------'
    auto mean1_axes = wrap_type<v0::Constant>();
    auto mean1 = wrap_type<v1::ReduceMean>({x, mean1_axes});

    // (x - ReduceMean(x, axes))
    // `-sub1------------------'
    auto sub1 = wrap_type<v1::Subtract>({x, mean1});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                     `---mean2----------'
    auto mean2_axes = wrap_type<v0::Constant>();
    auto mean2 = wrap_type<v1::ReduceMean>({x, mean2_axes});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `-sub2------------------'
    auto sub2 = wrap_type<v1::Subtract>({x, mean2});

    const auto reuseSub1OrNot = std::make_shared<Or>(OutputVector{sub1, sub2});
    const auto optionalConvert = ov::pass::pattern::optional<v0::Convert>(reuseSub1OrNot);

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //                 `---------------------power--'
    auto const_2 = wrap_type<v0::Constant>(value_is_equal_to<float>({2.0}));
    auto power = wrap_type<v1::Power>({optionalConvert, const_2});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    //     `---mean3--------------------------------'
    auto mean3_axes = wrap_type<v0::Constant>();
    auto mean3 = wrap_type<v1::ReduceMean>({power, mean3_axes});

    auto const_0_5 = wrap_type<v0::Constant>(value_is_equal_to<float>({0.5}));
    auto eps = wrap_type<v0::Constant>();
    // ------------------- OUTSIDE_SQRT ----------------------

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto power_sqrt_os = wrap_type<v1::Power>({mean3, const_0_5});
    auto sqrt_os = wrap_type<v0::Sqrt>({mean3});
    const auto powerOrSqrt_os = std::make_shared<Or>(OutputVector{power_sqrt_os, sqrt_os});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps
    // `----------------------------------------------Add---'
    auto add_eps_os = wrap_type<v1::Add>({powerOrSqrt_os, eps});

    // ------------------- INSIDE_SQRT ----------------------

    // (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps))
    // `-----------------------------------------------Add---'
    auto add_eps_is = wrap_type<v1::Add>({mean3, eps});

    // Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2))
    // `--Power--------------------------------------'
    auto power_sqrt_is = wrap_type<v1::Power>({add_eps_is, const_0_5});
    auto sqrt_is = wrap_type<v0::Sqrt>({add_eps_is});
    const auto powerOrSqrt_is = std::make_shared<Or>(OutputVector{power_sqrt_is, sqrt_is});

    auto outsideOrInside = std::make_shared<Or>(OutputVector{add_eps_os, powerOrSqrt_is});

    // Final Divide
    auto const_neg_1 = wrap_type<v0::Constant>(value_is_equal_to<float>({-1}));
    auto power_div = wrap_type<v1::Power>({outsideOrInside, const_neg_1});
    auto div = wrap_type<v1::Multiply>({sub1, power_div});

    auto div_alt = wrap_type<v1::Divide>({sub1, outsideOrInside});
    const auto powerMulOrDiv = std::make_shared<Or>(OutputVector{div, div_alt});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto exp_input = pattern_to_output.at(x);

        auto const_eps_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!op_util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto axes_1_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_3_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(mean3_axes).get_node_shared_ptr());

        if (!axes_1_node || !axes_3_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_3_value = axes_3_node->cast_vector<int64_t>();

        if (axes_1_value != axes_3_value) {
            return false;
        }
        if (pattern_to_output.count(mean2_axes)) {
            auto axes_2_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
            if (!axes_2_node) {
                return false;
            }
            auto axes_2_value = axes_2_node->cast_vector<int64_t>();
            if (axes_1_value != axes_2_value) {
                return false;
            }
        }

        ov::NodeVector nodes_to_copy_info({pattern_to_output.at(mean1).get_node_shared_ptr(),
                                           pattern_to_output.at(sub1).get_node_shared_ptr(),
                                           pattern_to_output.at(power).get_node_shared_ptr(),
                                           pattern_to_output.at(mean3).get_node_shared_ptr()});

        ov::op::MVNEpsMode mode;
        if (pattern_to_output.count(add_eps_os)) {
            mode = ov::op::MVNEpsMode::OUTSIDE_SQRT;
            nodes_to_copy_info.push_back(pattern_to_output.at(add_eps_os).get_node_shared_ptr());
            if (pattern_to_output.count(power_sqrt_os)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt_os).get_node_shared_ptr());
            } else if (pattern_to_output.count(sqrt_os)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(sqrt_os).get_node_shared_ptr());
            }
        } else if (pattern_to_output.count(powerOrSqrt_is)) {
            mode = ov::op::MVNEpsMode::INSIDE_SQRT;
            nodes_to_copy_info.push_back(pattern_to_output.at(add_eps_is).get_node_shared_ptr());
            if (pattern_to_output.count(power_sqrt_is)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(power_sqrt_is).get_node_shared_ptr());
            } else if (pattern_to_output.count(sqrt_is)) {
                nodes_to_copy_info.push_back(pattern_to_output.at(sqrt_is).get_node_shared_ptr());
            }
        } else {
            return false;
        }
        auto mvn = std::make_shared<v6::MVN>(exp_input, axes_1_node, true, eps_value, mode);

        if (pattern_to_output.count(mean2) && pattern_to_output.count(sub2)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(mean2).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(sub2).get_node_shared_ptr());
        }

        if (pattern_to_output.count(optionalConvert)) {
            auto cast = pattern_to_output.at(optionalConvert).get_node_shared_ptr();
            nodes_to_copy_info.push_back(cast);
        }

        if (pattern_to_output.count(div_alt)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(div_alt).get_node_shared_ptr());
        } else if (pattern_to_output.count(power_div) && pattern_to_output.count(div)) {
            nodes_to_copy_info.push_back(pattern_to_output.at(power_div).get_node_shared_ptr());
            nodes_to_copy_info.push_back(pattern_to_output.at(div).get_node_shared_ptr());
        }

        mvn->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(nodes_to_copy_info, mvn);
        ov::replace_node(m.get_match_root(), mvn);
        return true;
    };

    auto m = std::make_shared<Matcher>(powerMulOrDiv, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::MVNFusionWithConstantsInside::MVNFusionWithConstantsInside() {
    MATCHER_SCOPE(MVNFusionWithConstantsInside);
    // Detect MVN decomposition pattern:
    // (x - ReduceMean(x, axes)) * gamma / (Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2)) + eps) + beta
    auto x = any_input();

    // (x - ReduceMean(x, axes))^2
    //     `------mean1-------'
    auto mean1_axes = wrap_type<v0::Constant>();
    auto mean1 = wrap_type<v1::ReduceMean>({x, mean1_axes});

    // (x - ReduceMean(x, axes))^2
    // `-squared_difference------'
    auto squared_difference = wrap_type<v0::SquaredDifference>({x, mean1});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `---mean2--------------------------------'
    auto mean2_axes = wrap_type<v0::Constant>();
    auto mean2 = wrap_type<v1::ReduceMean>({squared_difference, mean2_axes});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    //         `------------------------------------------add--'
    auto eps = wrap_type<v0::Constant>();
    auto add_eps = wrap_type<v1::Add>({mean2, eps});

    // 1 / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `-power-------------------------------------------------'
    auto const_0_5 = wrap_type<v0::Constant>(value_is_equal_to<float>({-0.5}));
    auto power = wrap_type<v1::Power>({add_eps, const_0_5});

    // gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul1----------------------------------------------------'
    auto gamma = wrap_type<v0::Constant>();
    auto mul1 = wrap_type<v1::Multiply>({power, gamma});

    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---mul2--------------------------------------------------------'
    auto mul2 = wrap_type<v1::Multiply>({x, mul1});

    // ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) - beta
    // `-------------------mul3----------------------------------------------------------'
    auto mul3 = wrap_type<v1::Multiply>({mul1, mean1});

    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
    // `---sub-----------------------------------------------------------------------------------'
    auto beta = wrap_type<v0::Constant>();
    auto sub = wrap_type<v1::Subtract>({beta, mul3});

    // Final Add
    // x * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) +
    // beta - ReduceMean(x, axes) * gamma / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) =
    // gamma * (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps) + beta
    auto add = wrap_type<v1::Add>({mul2, sub});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(x);

        auto const_0_5_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(const_0_5).get_node_shared_ptr());
        auto const_gamma_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(gamma).get_node_shared_ptr());
        auto const_beta_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(beta).get_node_shared_ptr());
        auto const_eps_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(eps).get_node_shared_ptr());
        if (!const_0_5_node || !const_beta_node || !const_gamma_node || !const_eps_node) {
            return false;
        }

        float eps_value;
        bool valid_constant_values = op_util::has_constant_value<float>(const_0_5_node, -0.5) &&
                                     op_util::get_single_value(const_eps_node, eps_value);
        if (!valid_constant_values) {
            return false;
        }

        auto axes_1_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(mean1_axes).get_node_shared_ptr());
        auto axes_2_node = ov::as_type_ptr<v0::Constant>(pattern_to_output.at(mean2_axes).get_node_shared_ptr());
        if (!axes_1_node || !axes_2_node) {
            return false;
        }

        auto axes_1_value = axes_1_node->cast_vector<int64_t>();
        auto axes_2_value = axes_2_node->cast_vector<int64_t>();
        if (axes_1_value != axes_2_value) {
            return false;
        }

        auto mvn = std::make_shared<v6::MVN>(x_output, axes_1_node, true, eps_value, ov::op::MVNEpsMode::INSIDE_SQRT);
        auto mul_gamma = std::make_shared<v1::Multiply>(mvn, const_gamma_node);
        auto add_beta = std::make_shared<v1::Add>(mul_gamma, const_beta_node);

        ov::copy_runtime_info({pattern_to_output.at(mean1).get_node_shared_ptr(),
                               pattern_to_output.at(squared_difference).get_node_shared_ptr(),
                               pattern_to_output.at(add_eps).get_node_shared_ptr(),
                               pattern_to_output.at(power).get_node_shared_ptr(),
                               pattern_to_output.at(mul1).get_node_shared_ptr(),
                               pattern_to_output.at(mul2).get_node_shared_ptr(),
                               pattern_to_output.at(mul3).get_node_shared_ptr(),
                               pattern_to_output.at(sub).get_node_shared_ptr(),
                               pattern_to_output.at(add).get_node_shared_ptr()},
                              {mvn, const_gamma_node, mul_gamma, const_beta_node, add_beta});
        add_beta->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::replace_node(m.get_match_root(), add_beta);
        return true;
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
