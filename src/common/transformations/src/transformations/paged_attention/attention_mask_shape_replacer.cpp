// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/attention_mask_shape_replacer.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::has_static_rank;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;
namespace v3 = ov::op::v3;

ov::pass::AttentionMaskShapeReplacer::AttentionMaskShapeReplacer(const Output<Node>& input_source) {
    MATCHER_SCOPE(AttentionMaskShapeReplacer);

    auto attn_mask = wrap_type<v0::Parameter>(has_static_rank() && [](const Output<Node>& output) {
        return output.get_names().count("attention_mask") > 0;
    });
    auto shape_of = wrap_type<ov::op::util::ShapeOfBase>({attn_mask});
    auto gather = wrap_type<ov::op::util::GatherBase>({shape_of, any_input(), any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto gather_node = pattern_map.at(gather).get_node_shared_ptr();
        const auto indices_const = ov::util::get_constant_from_source(gather_node->input_value(1));
        if (!indices_const) {
            return false;
        }

        // Only the batch dimension (index 0) is guaranteed to coincide between attention_mask
        // and the input source regardless of their ranks, so the rewrite is limited to it.
        // This also keeps the same Gather index valid and avoids negative-index ambiguity.
        for (int64_t index : indices_const->cast_vector<int64_t>()) {
            if (index != 0) {
                return false;
            }
        }

        const auto old_shape_of = pattern_map.at(shape_of).get_node_shared_ptr();
        auto new_shape_of = std::make_shared<v3::ShapeOf>(input_source, old_shape_of->get_output_element_type(0));
        new_shape_of->set_friendly_name(old_shape_of->get_friendly_name());
        ov::copy_runtime_info(old_shape_of, new_shape_of);

        gather_node->input(0).replace_source_output(new_shape_of);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather, matcher_name);
    register_matcher(m, callback);
}
