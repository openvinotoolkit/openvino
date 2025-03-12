// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_scale_fusion.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "transformations/utils/gen_pattern.hpp"

namespace ov {
namespace pass {

SDPAScaleFusion::SDPAScaleFusion() {
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;

    auto q = makePattern(ov::Rank(4));
    auto k = makePattern(ov::Rank(4));
    auto v = makePattern(ov::Rank(4));
    auto mask = makePattern();
    auto sdpa_scale = makeConst({});
    auto scale_q = makePattern("[]") | makePattern("[1]");
    auto scale_k = makePattern("[]") | makePattern("[1]");

    auto scaled_q = optional<ov::op::v1::Multiply>({q, scale_q});
    auto scaled_k = optional<ov::op::v1::Multiply>({k, scale_k});
    auto sdpa_mask_scale =
        makePattern<ov::op::v13::ScaledDotProductAttention>({scaled_q, scaled_k, v, mask, sdpa_scale},
                                                            {{"causal", false}});
    auto sdpa_mask =
        makePattern<ov::op::v13::ScaledDotProductAttention>({scaled_q, scaled_k, v, mask}, {{"causal", false}});
    auto sdpa_simple =
        makePattern<ov::op::v13::ScaledDotProductAttention>({scaled_q, scaled_k, v}, {{"causal", false}});
    auto sdpa = sdpa_simple | sdpa_mask | sdpa_mask_scale;

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto sdpa = m.get_match_root();

        bool has_q_scale = pattern_map.count(scaled_q);
        bool has_k_scale = pattern_map.count(scaled_k);

        // Nothing to do
        if (!has_q_scale && !has_k_scale)
            return false;

        auto prev_scale_value = 1.0f;
        auto scale_q_value = 1.0f;
        auto scale_k_value = 1.0f;
        auto scale_et = sdpa->get_output_element_type(0);

        Output<ov::Node> q_input = sdpa->get_input_source_output(0);
        Output<ov::Node> k_input = sdpa->get_input_source_output(1);

        std::shared_ptr<ov::Node> scale_q_node = nullptr;
        std::shared_ptr<ov::Node> scale_k_node = nullptr;

        if (pattern_map.find(sdpa_scale) != pattern_map.end()) {
            auto prev_scale_node =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(sdpa_scale).get_node_shared_ptr());
            prev_scale_value = prev_scale_node->cast_vector<float>()[0];
            scale_et = prev_scale_node->get_output_element_type(0);
        } else {
            auto head_size = q_input.get_partial_shape()[3];
            if (head_size.is_dynamic())
                return false;

            prev_scale_value = 1.0f / std::sqrt(static_cast<float>(head_size.get_length()));
        }

        // Extract scalar scale values for Q and K if those are constant and set new inputs for SDPA
        if (has_q_scale) {
            scale_q_node = pattern_map.at(scale_q).get_node_shared_ptr();
            if (pattern_map.at(q).get_element_type() == q_input.get_element_type()) {
                if (ov::is_type<ov::op::v0::Constant>(scale_q_node)) {
                    scale_q_value = ov::as_type_ptr<ov::op::v0::Constant>(scale_q_node)->cast_vector<float>()[0];
                    q_input = pattern_map.at(q);
                }
            } else {
                has_q_scale = false;
            }
        }
        if (has_k_scale) {
            scale_k_node = pattern_map.at(scale_k).get_node_shared_ptr();
            if (pattern_map.at(k).get_element_type() == k_input.get_element_type()) {
                if (ov::is_type<ov::op::v0::Constant>(scale_k_node)) {
                    scale_k_value = ov::as_type_ptr<ov::op::v0::Constant>(scale_k_node)->cast_vector<float>()[0];
                    k_input = pattern_map.at(k);
                }
            } else {
                has_k_scale = false;
            }
        }

        if (!has_q_scale && !has_k_scale)
            return false;

        Output<ov::Node> new_scale_node;
        auto new_scale_val = prev_scale_value * scale_q_value * scale_k_value;
        // If new scale is 1 and we have non-constant scale node for either Q or K, then we can make it a scale of SDPA
        if (new_scale_val == 1.0f) {
            if (has_q_scale && !ov::is_type<ov::op::v0::Constant>(scale_q_node)) {
                new_scale_node = pattern_map.at(scale_q);
                q_input = pattern_map.at(q);
            } else if (has_k_scale && !ov::is_type<ov::op::v0::Constant>(scale_k_node)) {
                new_scale_node = pattern_map.at(scale_k);
                k_input = pattern_map.at(k);
            } else {
                new_scale_node = ov::op::v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{new_scale_val});
            }
        } else {
            new_scale_node = ov::op::v0::Constant::create(scale_et, ov::Shape{}, std::vector<float>{new_scale_val});
        }

        OutputVector new_inputs = {q_input, k_input, pattern_map.at(v)};
        if (pattern_map.find(mask) != pattern_map.end()) {
            new_inputs.push_back(pattern_map.at(mask));
        } else {
            new_inputs.push_back(
                ov::op::v0::Constant::create(new_scale_node.get_element_type(), ov::Shape{}, std::vector<float>{0.0f}));
        }

        new_inputs.push_back(new_scale_node);

        auto new_sdpa = sdpa->clone_with_new_inputs(new_inputs);
        new_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(sdpa, new_sdpa);
        ov::replace_node(sdpa, new_sdpa);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa, "SDPAScaleFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
