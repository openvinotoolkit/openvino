// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatterupdate_reshape_concat_reshape_sdpa_fusion.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "transformations/utils/utils.hpp"

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include <openvino/op/variadic_split.hpp>

namespace ov::intel_gpu {
ScatterUpdateReshapeConcatReshapeSDPAFusion::ScatterUpdateReshapeConcatReshapeSDPAFusion() {
    using namespace ov::pass::pattern;
   
    auto input_a_m = any_input();
    auto input_attn_mask_m = any_input();
    auto input_scale_m = any_input();

    auto past_param_key_m = wrap_type<ov::op::v0::Parameter>();
    auto indices_key_m = any_input();
    auto updates_key_m = any_input();
    auto axis_key_m = wrap_type<ov::op::v0::Constant>();

    auto scatter_update_key_m = wrap_type<ov::op::v3::ScatterUpdate>({past_param_key_m, indices_key_m, updates_key_m, axis_key_m});

    auto vsplit_axis_key_m = any_input();
    auto vsplit_lengths_key_m = any_input();
    auto variadic_split_key_m = wrap_type<ov::op::v1::VariadicSplit>(
        {scatter_update_key_m, vsplit_axis_key_m, vsplit_lengths_key_m});

    auto scatter_or_vsplit_key_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{scatter_update_key_m, variadic_split_key_m});

    auto reshape1_pattern_key_m = any_input();
    auto reshape_4d_to_5d_key_m = wrap_type<ov::op::v1::Reshape>({scatter_or_vsplit_key_m, reshape1_pattern_key_m});

    auto concat_key_m = wrap_type<ov::op::v0::Concat>({reshape_4d_to_5d_key_m, reshape_4d_to_5d_key_m, reshape_4d_to_5d_key_m, reshape_4d_to_5d_key_m});

    auto reshape2_pattern_key_m = any_input();
    auto reshape_5d_to_4d_key_m = wrap_type<ov::op::v1::Reshape>({concat_key_m, reshape2_pattern_key_m});

    auto past_param_value_m = wrap_type<ov::op::v0::Parameter>();
    auto indices_value_m = any_input();
    auto updates_value_m = any_input();
    auto axis_value_m = wrap_type<ov::op::v0::Constant>();

    auto scatter_update_value_m = wrap_type<ov::op::v3::ScatterUpdate>({past_param_value_m, indices_value_m, updates_value_m, axis_value_m});

    auto vsplit_axis_value_m = any_input();
    auto vsplit_lengths_value_m = any_input();
    auto variadic_split_value_m = wrap_type<ov::op::v1::VariadicSplit>(
        { scatter_update_value_m, vsplit_axis_value_m, vsplit_lengths_value_m });

    auto scatter_or_vsplit_value_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{ scatter_update_value_m, variadic_split_value_m });

    auto reshape1_pattern_value_m = any_input();
    auto reshape_4d_to_5d_value_m = wrap_type<ov::op::v1::Reshape>({scatter_or_vsplit_value_m, reshape1_pattern_value_m});

    auto concat_value_m = wrap_type<ov::op::v0::Concat>({reshape_4d_to_5d_value_m, reshape_4d_to_5d_value_m, reshape_4d_to_5d_value_m, reshape_4d_to_5d_value_m});

    auto reshape2_pattern_value_m = any_input();
    auto reshape_5d_to_4d_value_m = wrap_type<ov::op::v1::Reshape>({concat_value_m, reshape2_pattern_value_m});

    auto sdpa_without_attn_mask_m = wrap_type<op::SDPA>({input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m});
    auto sdpa_with_attn_mask_m =
        wrap_type<op::SDPA>({input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m, any_input()});
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<op::SDPA>({input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m, input_attn_mask_m, input_scale_m});

    
    auto sdpa_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        std::shared_ptr<ov::intel_gpu::op::SDPA> sdpa;
        sdpa = std::dynamic_pointer_cast<ov::intel_gpu::op::SDPA>(
            pattern_map.at(sdpa_m).get_node_shared_ptr());
        const auto past_param_key =
            std::dynamic_pointer_cast<ov::op::v0::Parameter>(pattern_map.at(past_param_key_m).get_node_shared_ptr());
        const auto past_param_value =
            std::dynamic_pointer_cast<ov::op::v0::Parameter>(pattern_map.at(past_param_value_m).get_node_shared_ptr());


        OutputVector data_inputs;

        if (pattern_map.count(scatter_update_key_m) > 0) {
            const auto scatter_update_key =
                std::dynamic_pointer_cast<ov::op::v3::ScatterUpdate>(pattern_map.at(scatter_update_key_m).get_node_shared_ptr());
        } if (pattern_map.count(variadic_split_key_m) > 0) {
            const auto variadic_split_key =
                std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(pattern_map.at(variadic_split_key_m).get_node_shared_ptr());
            sdpa->input(1).replace_source_output(variadic_split_key->output(1));
        }
        if (pattern_map.count(scatter_update_value_m) > 0) {
            const auto scatter_update_value =
                std::dynamic_pointer_cast<ov::op::v3::ScatterUpdate>(pattern_map.at(scatter_update_value_m).get_node_shared_ptr());
        } if (pattern_map.count(variadic_split_value_m) > 0) {
            const auto variadic_split_value =
                std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(pattern_map.at(variadic_split_value_m).get_node_shared_ptr());
            sdpa->input(2).replace_source_output(variadic_split_value->output(1));

        }
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "ScatterUpdateReshapeConcatReshapeSDPAFusion");

    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu