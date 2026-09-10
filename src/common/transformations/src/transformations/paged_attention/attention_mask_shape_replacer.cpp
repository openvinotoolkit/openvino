// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/attention_mask_shape_replacer.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
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
    auto concat = wrap_type<v0::Concat>({gather, any_input(), any_input()});
    auto broadcast = wrap_type<v3::Broadcast>({any_input(), concat});
    auto position_ids = wrap_type<v0::Parameter>(has_static_rank() && [](const Output<Node>& output) {
        return output.get_names().count("position_ids") > 0;
    });
    auto position_ids_unsqueeze_0 = ov::pass::pattern::optional<v0::Unsqueeze>({position_ids, any_input()});
    auto position_ids_unsqueeze_1 = ov::pass::pattern::optional<v0::Unsqueeze>({position_ids_unsqueeze_0, any_input()});
    auto position_ids_convert = ov::pass::pattern::optional<v0::Convert>({position_ids_unsqueeze_1});
    auto matmul = wrap_type<v0::MatMul>({broadcast, position_ids_convert});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        // The batch dimension is read from ShapeOf(input_source) at index 0, so the source must
        // expose it: a static rank of at least 1 is required for the rewrite to stay valid.
        const auto source_rank = input_source.get_partial_shape().rank();
        if (source_rank.is_dynamic() || source_rank.get_length() < 1) {
            return false;
        }

        auto gather_node = pattern_map.at(gather).get_node_shared_ptr();
        const auto indices_const = ov::util::get_constant_from_source(gather_node->input_value(1));
        if (!indices_const) {
            return false;
        }

        // The shape produced by ShapeOf is a 1D vector, so the Gather axis must select it (0, or the
        // equivalent -1). Rewiring a Gather with an unexpected or dynamic axis could change behavior.
        const auto axis_const = ov::util::get_constant_from_source(gather_node->input_value(2));
        if (!axis_const) {
            return false;
        }
        for (int64_t axis : axis_const->cast_vector<int64_t>()) {
            if (axis != 0 && axis != -1) {
                return false;
            }
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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul, matcher_name);
    register_matcher(m, callback);
}
