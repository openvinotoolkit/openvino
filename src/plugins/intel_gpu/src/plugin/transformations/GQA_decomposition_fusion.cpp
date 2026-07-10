// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "GQA_Decomposition_fusion.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov::intel_gpu {
    GQADecompositionfusion::GQADecompositionfusion() {
        using namespace ov::pass::pattern;

        auto reshape_4d_to_5d = [](const ov::Output<ov::Node>& output) {
            auto in_ps = output.get_node()->get_input_partial_shape(0);
            auto out_ps = output.get_node()->get_output_partial_shape(0);
            return in_ps.rank().is_static() && out_ps.rank().is_static() && in_ps.size() == 4 && out_ps.size() == 5;
        };

        auto reshape_5d_to_4d = [](const ov::Output<ov::Node>& output) {
            auto in_ps = output.get_node()->get_input_partial_shape(0);
            auto out_ps = output.get_node()->get_output_partial_shape(0);
            return in_ps.rank().is_static() && out_ps.rank().is_static() && in_ps.size() == 5 && out_ps.size() == 4;
        };

        auto input_a_m = any_input();
        auto input_attn_mask_m = any_input();
        auto input_scale_m = any_input();

        auto in_key = any_input();
        auto reshape1_pattern_key_m = any_input();
        auto reshape_4d_to_5d_key_m = wrap_type<ov::op::v1::Reshape>({ in_key, reshape1_pattern_key_m }, reshape_4d_to_5d);

        auto concat_key_m = wrap_type<ov::op::v0::Concat>({ reshape_4d_to_5d_key_m, reshape_4d_to_5d_key_m, reshape_4d_to_5d_key_m, reshape_4d_to_5d_key_m });

        auto reshape2_pattern_key_m = any_input();
        auto reshape_5d_to_4d_key_m = wrap_type<ov::op::v1::Reshape>({ concat_key_m, reshape2_pattern_key_m }, reshape_5d_to_4d);

        auto in_value = any_input();
        auto reshape1_pattern_value_m = any_input();
        auto reshape_4d_to_5d_value_m = wrap_type<ov::op::v1::Reshape>({ in_value, reshape1_pattern_value_m });

        auto concat_value_m = wrap_type<ov::op::v0::Concat>({ reshape_4d_to_5d_value_m, reshape_4d_to_5d_value_m, reshape_4d_to_5d_value_m, reshape_4d_to_5d_value_m });

        auto reshape2_pattern_value_m = any_input();
        auto reshape_5d_to_4d_value_m = wrap_type<ov::op::v1::Reshape>({ concat_value_m, reshape2_pattern_value_m });

        auto sdpa_without_attn_mask_m = wrap_type<op::SDPA>({ input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m });
        auto sdpa_with_attn_mask_m =
            wrap_type<op::SDPA>({ input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m, any_input() });
        auto sdpa_with_attn_mask_and_scale_m =
            wrap_type<op::SDPA>({ input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m, input_attn_mask_m, input_scale_m });

        auto sdpa_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{ sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m });

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();

            auto sdpa = ov::as_type_ptr<op::SDPA>(m.get_match_root());

            if (pattern_map.count(in_key) > 0) {
                const auto key = pattern_map.at(in_key).get_node_shared_ptr();
                sdpa->input(1).replace_source_output(key->output(1));
            }

            if (pattern_map.count(in_value) > 0) {
                const auto value = pattern_map.at(in_value).get_node_shared_ptr();
                sdpa->input(2).replace_source_output(value->output(1));

            }
            return true;
            };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "ScatterUpdateReshapeConcatReshapeSDPAFusion");

        this->register_matcher(m, callback);
    }

}  // namespace ov::intel_gpu