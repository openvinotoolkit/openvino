// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/gen_pattern.hpp"

namespace ov {
namespace pass {

SDPAFusion::SDPAFusion() {
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;

    auto q = makePattern(ov::Rank(4));
    auto k = makePattern(ov::Rank(4));
    auto v = makePattern(ov::Rank(4));
    auto mask = makePattern();

    auto k_transpose_order = pattern::wrap_type<ov::op::v0::Constant>([](const Output<Node>& node) {
        auto axis_order = ov::as_type_ptr<ov::op::v0::Constant>(node.get_node_shared_ptr())->cast_vector<int64_t>();
        return axis_order == std::vector<int64_t>{0, 1, 3, 2};
    });

    auto k_t = pattern::wrap_type<ov::op::v1::Transpose>({k, k_transpose_order});
    auto qk_nn = makePattern<ov::op::v0::MatMul>({q, k_t}, {{"transpose_a", false}, {"transpose_b", false}});
    auto qk_nt = makePattern<ov::op::v0::MatMul>({q, k}, {{"transpose_a", false}, {"transpose_b", true}});
    auto qk = qk_nt | qk_nn;
    auto optional_add_mask = optional<ov::op::v1::Add>({qk, mask});
    auto softmax = makePattern<ov::op::v8::Softmax>({optional_add_mask}, {{"axis", "-1"}});
    auto qkv = makePattern<ov::op::v0::MatMul>({softmax, v}, {{"transpose_a", false}, {"transpose_b", false}});

    auto valid_qk_shapes = [](const std::shared_ptr<ov::op::v0::MatMul>& qk_matmul) {
        auto q_pshape = qk_matmul->get_input_partial_shape(0);
        auto k_pshape = qk_matmul->get_input_partial_shape(1);

        const size_t q_head_size_idx = 3;
        const size_t k_head_size_idx = qk_matmul->get_transpose_b() ? 3 : 2;

        return q_pshape.size() == 4 && k_pshape.size() == 4 && q_pshape[q_head_size_idx].is_static() &&
               k_pshape[k_head_size_idx].is_static() &&
               q_pshape[q_head_size_idx].get_length() == k_pshape[k_head_size_idx].get_length();
    };

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto q_node = pattern_map.at(q);
        auto k_node = pattern_map.at(k);
        auto v_node = pattern_map.at(v);

        if (!valid_qk_shapes(ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(qk).get_node_shared_ptr()))) {
            return false;
        }

        if (pattern_map.at(qk).get_target_inputs().size() > 1 ||
            pattern_map.at(softmax).get_target_inputs().size() > 1) {
            return false;
        }
        if (pattern_map.count(optional_add_mask) && (pattern_map.at(optional_add_mask).get_target_inputs().size() > 1 ||
                                                     pattern_map.at(mask).get_partial_shape().size() > 4)) {
            return false;
        }

        Output<ov::Node> mask_value;
        Output<ov::Node> mask_input;
        if (pattern_map.find(optional_add_mask) != pattern_map.end()) {
            mask_value = pattern_map.at(mask);
        } else {
            mask_value = ov::op::v0::Constant::create(q_node.get_element_type(), ov::Shape{}, std::vector<float>{0});
        }

        if (mask_value.get_partial_shape().size() > 4) {
            return false;
        }

        if (mask_value.get_partial_shape().rank() == 0 || mask_value.get_partial_shape().rank() == 4) {
            mask_input = mask_value;
        } else {
            size_t rank_diff = q_node.get_partial_shape().size() - mask_value.get_partial_shape().size();
            std::vector<int64_t> axes(rank_diff);
            std::iota(axes.begin(), axes.end(), 0);
            mask_input = std::make_shared<ov::op::v0::Unsqueeze>(
                mask_value,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank_diff}, axes));
        }

        std::shared_ptr<ov::Node> scale_node =
            ov::op::v0::Constant::create(q_node.get_element_type(), ov::Shape{}, std::vector<float>{1.0f});

        std::shared_ptr<ov::Node> sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_node,
                                                                                                  k_node,
                                                                                                  v_node,
                                                                                                  mask_input,
                                                                                                  scale_node,
                                                                                                  false);

        sdpa->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);
        ov::replace_node(m.get_match_root(), sdpa);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(qkv, "SDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
