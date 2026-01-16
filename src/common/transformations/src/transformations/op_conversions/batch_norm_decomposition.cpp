// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/batch_norm_decomposition.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v5 = ov::op::v5;

namespace ov::pass {

BatchNormDecomposition::BatchNormDecomposition() {
    MATCHER_SCOPE(BatchNormDecomposition);
    auto bn_1 = pattern::wrap_type<v0::BatchNormInference>({pattern::any_input(),
                                                            pattern::any_input(),
                                                            pattern::any_input(pattern::has_static_rank()),
                                                            pattern::any_input(),
                                                            pattern::any_input()});
    auto bn_5 = pattern::wrap_type<v5::BatchNormInference>({pattern::any_input(pattern::has_static_rank()),
                                                            pattern::any_input(),
                                                            pattern::any_input(),
                                                            pattern::any_input(),
                                                            pattern::any_input()});
    auto bn = std::make_shared<pattern::op::Or>(OutputVector{bn_1, bn_5});

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto m_bn = m.get_match_root();
        Output<Node> m_input, m_gamma, m_beta, m_mean, m_var;
        double eps;
        if (auto m_bn_v1 = ov::as_type_ptr<v0::BatchNormInference>(m_bn)) {
            m_gamma = m_bn_v1->input_value(0);
            m_beta = m_bn_v1->input_value(1);
            m_input = m_bn_v1->input_value(2);
            m_mean = m_bn_v1->input_value(3);
            m_var = m_bn_v1->input_value(4);
            eps = m_bn_v1->get_eps_value();
        } else if (auto m_bn_v5 = ov::as_type_ptr<v5::BatchNormInference>(m_bn)) {
            m_input = m_bn_v5->input_value(0);
            m_gamma = m_bn_v5->input_value(1);
            m_beta = m_bn_v5->input_value(2);
            m_mean = m_bn_v5->input_value(3);
            m_var = m_bn_v5->input_value(4);
            eps = m_bn_v5->get_eps_value();
        } else {
            return false;
        }

        const auto& input_type = m_input.get_element_type();
        // scale_add = variance + eps
        auto scale_add = std::make_shared<v1::Add>(m_var, v0::Constant::create(input_type, Shape{}, {eps}));
        // scale = sqrt(variance + eps)
        auto scale = std::make_shared<v0::Sqrt>(scale_add);
        // Divide `gamma` by `sqrt(variance + eps)`
        auto gamma_div_scale = std::make_shared<v1::Divide>(m_gamma, scale);

        int64_t dims_to_add = m_input.get_partial_shape().rank().get_length() - 2;
        const auto one = v0::Constant::create(element::i64, Shape{1}, {1});
        const auto tail_shape_rank = v0::Constant::create(element::i64, Shape{1}, {dims_to_add});
        const auto tail_shape = std::make_shared<v3::Broadcast>(one, tail_shape_rank);
        const auto C_dim = std::make_shared<v3::ShapeOf>(m_gamma);
        // create new shape [1, C, 1, 1, ...]
        const auto new_shape = std::make_shared<v0::Concat>(OutputVector{one, C_dim, tail_shape}, 0);

        std::shared_ptr<Node> gamma_div_scale_aligned = std::make_shared<v1::Reshape>(gamma_div_scale, new_shape, true);
        std::shared_ptr<Node> beta_aligned = std::make_shared<v1::Reshape>(m_beta, new_shape, true);
        std::shared_ptr<Node> mean_aligned = std::make_shared<v1::Reshape>(m_mean, new_shape, true);
        auto mul_const = v0::Constant::create(mean_aligned->get_output_element_type(0), Shape{}, {-1});
        std::shared_ptr<Node> mean_negative = std::make_shared<v1::Multiply>(mean_aligned, mul_const);

        if (auto constant = ov::util::get_constant_from_source(beta_aligned))
            beta_aligned = constant;
        if (auto constant = ov::util::get_constant_from_source(mean_negative))
            mean_negative = constant;
        if (auto constant = ov::util::get_constant_from_source(gamma_div_scale_aligned))
            gamma_div_scale_aligned = constant;

        // input_sub_mean = input + mean * -1
        auto input_sub_mean = register_new_node<v1::Add>(m_input, mean_negative);
        // Multiply  `input - mean` and `gamma / sqrt(variance + eps)`
        auto mul = register_new_node<v1::Multiply>(input_sub_mean, gamma_div_scale_aligned);
        // Add `(input - mean) * gamma / sqrt(variance + eps)` and `beta`
        auto add = register_new_node<v1::Add>(mul, beta_aligned);

        add->set_friendly_name(m_bn->get_friendly_name());

        copy_runtime_info(m_bn,
                          {scale_add,
                           scale,
                           gamma_div_scale,
                           gamma_div_scale_aligned,
                           beta_aligned,
                           input_sub_mean,
                           mul,
                           add,
                           mean_negative,
                           mean_aligned,
                           new_shape,
                           tail_shape,
                           tail_shape_rank,
                           one,
                           mul_const,
                           C_dim});

        replace_node(m_bn, add);

        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(bn, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
