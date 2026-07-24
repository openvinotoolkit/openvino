// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/rms_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::pass {

RMSFusion::RMSFusion(bool force_tail_convert, bool enable_without_gamma) {
    // Detect RMS decomposition pattern
    //  x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
    auto x = pattern::any_input();

    // x^2
    auto const_power = pattern::wrap_type<v0::Constant>(pattern::value_matches("2"));
    auto const_power_convert = pattern::optional<v0::Convert>(const_power);
    auto power = pattern::wrap_type<v1::Power>({x, const_power_convert});

    // x^2 via Multiply(x, x) — used by TFLite RMS norm decomposition
    auto mul_square = pattern::wrap_type<v1::Multiply>({x, x});

    // Either Power(x, 2) or Multiply(x, x)
    auto square = std::make_shared<pattern::op::Or>(OutputVector{power, mul_square});

    // ReduceMean(x^2,axes)
    auto mean_axes = pattern::wrap_type<v0::Constant>([](const ov::Output<ov::Node>& output) {
        auto const_node = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
        if (!const_node) {
            return false;
        }
        const auto& axes_shape = const_node->get_output_shape(0);
        const auto num_elems = ov::shape_size(axes_shape);
        // RMS fusion is only valid when ReduceMean has exactly one axis.
        return num_elems == 1;
    });
    auto mean = pattern::wrap_type<v1::ReduceMean>({square, mean_axes});

    // ReduceMean(x^2,axes)+eps
    auto eps = pattern::wrap_type<v0::Constant>();
    auto eps_convert = pattern::optional<v0::Convert>(eps);
    auto add_eps = pattern::wrap_type<v1::Add>({mean, eps_convert});

    // Optional Reshape between add_eps and sqrt/rsqrt (e.g., TFLite decomposition with keepdims=false)
    auto add_eps_opt_reshape = pattern::optional<v1::Reshape>({add_eps, pattern::any_input()});

    // Sqrt(ReduceMean(x^2,axes)+eps)
    auto sqrt = pattern::wrap_type<v0::Sqrt>({add_eps_opt_reshape});

    // 1/Sqrt(ReduceMean(x^2,axes)+eps)
    auto const_pow = pattern::wrap_type<v0::Constant>(pattern::value_matches("-1"));
    auto const_pow_convert = pattern::optional<v0::Convert>(const_pow);
    auto pow = pattern::wrap_type<v1::Power>({sqrt, const_pow_convert});

    auto const_div = pattern::wrap_type<v0::Constant>(pattern::value_matches("1"));
    auto const_div_convert = pattern::optional<v0::Convert>(const_div);
    auto div = pattern::wrap_type<v1::Divide>({const_div_convert, sqrt});

    // Power(ReduceMean(x^2,axes)+eps, -0.5) — direct rsqrt without Sqrt node
    auto const_neg_half = pattern::wrap_type<v0::Constant>(pattern::value_matches("-0.5"));
    auto const_neg_half_convert = pattern::optional<v0::Convert>(const_neg_half);
    auto pow_direct = pattern::wrap_type<v1::Power>({add_eps_opt_reshape, const_neg_half_convert});

    std::shared_ptr<pattern::op::Or> div_or_pow = std::make_shared<pattern::op::Or>(OutputVector{div, pow, pow_direct});

    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
    auto mul1 = pattern::wrap_type<v1::Multiply>({x, div_or_pow});

    // x / Sqrt(ReduceMean(x^2,axes)+eps)
    auto div_x = pattern::wrap_type<v1::Divide>({x, sqrt});
    auto mul_or_div = std::make_shared<pattern::op::Or>(OutputVector{mul1, div_x});

    // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma (gamma is constant)
    auto gamma = pattern::wrap_type<v0::Constant>();
    auto gamma_convert = pattern::optional<v0::Convert>(gamma);

    std::shared_ptr<ov::Node> rms_mul;
    if (enable_without_gamma) {
        // When enable_without_gamma is true, the trailing gamma Multiply is optional:
        // - If present (gamma is Constant): fuse gamma into RMS
        // - If absent: create RMS without gamma (e.g., Gemma v_norm, LTX-Video)
        // Requires BackwardGraphRewrite to ensure gamma Multiply is visited first.
        rms_mul = pattern::optional<v1::Multiply>({mul_or_div, gamma_convert});
    } else {
        rms_mul = pattern::wrap_type<v1::Multiply>({gamma_convert, mul_or_div});
    }

    std::shared_ptr<ov::Node> comp = rms_mul;
    if (force_tail_convert) {
        // compress RMS result
        comp = pattern::wrap_type<v0::Convert>({rms_mul});
    }

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }

        const auto& x_output = pattern_map.at(x);

        auto const_eps_node = ov::as_type_ptr<v0::Constant>(pattern_map.at(eps).get_node_shared_ptr());
        float eps_value;
        if (!op_util::get_single_value(const_eps_node, eps_value)) {
            return false;
        }

        auto mul_or_div_node = pattern_map.at(mul_or_div).get_node_shared_ptr();
        bool elementwise_affine = pattern_map.count(rms_mul);

        std::shared_ptr<ov::Node> gamma_node;
        if (elementwise_affine) {
            gamma_node = pattern_map.at(gamma).get_node_shared_ptr();
            if (pattern_map.count(gamma_convert)) {
                gamma_node = pattern_map.at(gamma_convert).get_node_shared_ptr();
            }
        }

        const auto& mean_node = pattern_map.at(mean).get_node_shared_ptr();
        const auto& axes = pattern_map.at(mean_axes).get_node_shared_ptr();
        auto axes_constant = ov::as_type_ptr<v0::Constant>(axes);
        auto axes_val = axes_constant->cast_vector<int64_t>();
        // allow last dimension only
        if ((axes_val[0] != -1) &&
            (axes_val[0] != (static_cast<int64_t>(mean_node->get_input_partial_shape(0).size()) - 1))) {
            return false;
        }

        auto output_type = elementwise_affine ? m.get_match_root()->get_output_element_type(0)
                                              : mul_or_div_node->get_output_element_type(0);
        std::shared_ptr<ov::op::internal::RMS> rms =
            elementwise_affine ? std::make_shared<ov::op::internal::RMS>(x_output, gamma_node, eps_value, output_type)
                               : std::make_shared<ov::op::internal::RMS>(x_output, eps_value, output_type);
        if (elementwise_affine) {
            rms->set_friendly_name(m.get_match_root()->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), rms);
            ov::replace_node(m.get_match_root(), rms);
        } else {
            rms->set_friendly_name(mul_or_div_node->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), rms);
            ov::replace_node(mul_or_div_node, rms);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(comp, "RMSFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
