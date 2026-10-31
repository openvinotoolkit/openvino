// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "l2_norm_to_rms.hpp"

#include <cmath>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

L2NormToRMS::L2NormToRMS(size_t max_work_group_size) {
    using namespace ov::pass::pattern;

    auto x = any_input();
    auto x_convert = optional<ov::op::v0::Convert>({x});
    auto power = wrap_type<ov::op::v1::Power>({x_convert, wrap_type<ov::op::v0::Constant>(value_matches("2"))});
    auto axes = wrap_type<ov::op::v0::Constant>();
    auto sum = wrap_type<ov::op::v1::ReduceSum>({power, axes}, consumers_count(1));
    auto eps = wrap_type<ov::op::v0::Constant>();
    auto add_eps = wrap_type<ov::op::v1::Add>({sum, eps}, consumers_count(1));
    auto sqrt = wrap_type<ov::op::v0::Sqrt>({add_eps}, consumers_count(1));
    auto div = wrap_type<ov::op::v1::Divide>({x_convert, sqrt}, consumers_count(1));
    auto gamma = any_input();
    auto mul = wrap_type<ov::op::v1::Multiply>({div, gamma});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        auto sum_node = ov::as_type_ptr<ov::op::v1::ReduceSum>(pm.at(sum).get_node_shared_ptr());
        if (!sum_node || !sum_node->get_keep_dims() || transformation_callback(sum_node)) {
            return false;
        }

        // Only the last axis, and only a static one: N goes into the rewritten gamma and eps.
        const auto& in_shape = sum_node->get_input_partial_shape(0);
        if (in_shape.rank().is_dynamic() || in_shape[in_shape.size() - 1].is_dynamic()) {
            return false;
        }
        const auto rank = static_cast<int64_t>(in_shape.size());
        auto axes_values = ov::as_type_ptr<ov::op::v0::Constant>(pm.at(axes).get_node_shared_ptr())->cast_vector<int64_t>();
        if (axes_values.size() != 1 || (axes_values[0] != -1 && axes_values[0] != rank - 1)) {
            return false;
        }
        const auto n = static_cast<size_t>(in_shape[rank - 1].get_length());

        // Same limit RMSFusion applies on GPU: the kernel handles a row with one work group.
        const size_t vec_size = 8;
        if (n / vec_size > max_work_group_size) {
            return false;
        }

        const auto& gamma_out = pm.at(gamma);
        const auto& gamma_shape = gamma_out.get_partial_shape();
        if (gamma_shape.is_dynamic() || ov::shape_size(gamma_shape.to_shape()) != n) {
            return false;
        }

        float eps_value = 0.f;
        if (!ov::op::util::get_single_value(ov::as_type_ptr<ov::op::v0::Constant>(pm.at(eps).get_node_shared_ptr()), eps_value)) {
            return false;
        }

        const auto x_out = pm.at(x);
        // Nodes are visited in topological order, so the match lands on the Multiply before its Convert is
        // seen; take a lone trailing Convert in as well so RMS writes the converted type directly.
        std::shared_ptr<ov::Node> root = pm.at(mul).get_node_shared_ptr();
        const auto& mul_users = root->get_output_target_inputs(0);
        if (mul_users.size() == 1) {
            if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(mul_users.begin()->get_node()->shared_from_this())) {
                root = convert;
            }
        }
        const auto output_type = root->get_output_element_type(0);

        // x / sqrt(sum + eps) * gamma == x / sqrt(mean + eps / n) * (gamma / sqrt(n))
        auto inv_sqrt_n = ov::op::v0::Constant::create(gamma_out.get_element_type(), {}, {1.0 / std::sqrt(static_cast<double>(n))});
        auto scaled_gamma = std::make_shared<ov::op::v1::Multiply>(gamma_out, inv_sqrt_n);
        auto rms = std::make_shared<ov::op::internal::RMS>(x_out, scaled_gamma, eps_value / static_cast<float>(n), output_type);

        rms->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), {scaled_gamma, rms});
        ov::replace_node(root, rms);
        return true;
    };

    auto m = std::make_shared<Matcher>(mul, "L2NormToRMS");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
