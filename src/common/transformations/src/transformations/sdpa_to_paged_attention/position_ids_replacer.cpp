// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/position_ids_replacer.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"
// TODO: Instead of using the following transformation that matches quite a specific place in a model graph in case when
// position_ids parameter is missing, consider replacing always existing attention_mask parameter with a sub-graph using
// a new slot_mapping parameter.
ov::pass::PositionIDsReplacer::PositionIDsReplacer(const Output<Node>& position_ids) {
    MATCHER_SCOPE(PositionIDsReplacer);

    auto input_ids = ov::pass::pattern::any_input();
    auto input_embed = ov::pass::pattern::wrap_type<ov::op::v8::Gather>({ov::pass::pattern::any_input(), input_ids, ov::pass::pattern::any_input()});

    auto position_ids_pattern = ov::pass::pattern::any_input();
    auto offset = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add_offset = ov::pass::pattern::wrap_type<ov::op::v1::Add>({position_ids_pattern, offset});
    auto convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({add_offset});
    auto position_embed = ov::pass::pattern::wrap_type<ov::op::v8::Gather>({ov::pass::pattern::any_input(), convert, ov::pass::pattern::any_input()});

    auto mul = ov::pass::pattern::optional<ov::op::v0::MatMul>({input_embed, ov::pass::pattern::any_input()});

    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({mul, position_embed});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        replace_node(pattern_map.at(position_ids_pattern).get_node_shared_ptr(), position_ids.get_node_shared_ptr());
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PositionIDsReplacerQwen::PositionIDsReplacerQwen(const Output<Node>& position_ids) {
    MATCHER_SCOPE(PositionIDsReplacerQwen);

    // total seq len:
    auto p_max_context_len = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>();
    auto p_opt_convert = ov::pass::pattern::optional<ov::op::v0::Convert>(p_max_context_len);
    auto p_opt_reshape = ov::pass::pattern::optional<ov::op::v1::Reshape>({p_opt_convert, ov::pass::pattern::any_input()});

    // current seq len:
    // it might be present in 2 different ways:
    // input_ids -> unsqueeze -> reshape -> convert -> shape_of -> gather
    // QKV -> variadic_split(Q or K) -> rope Q/K -> shape_of -> gather
    // Probably we can use the symbols to re-use one of these ways.
    // Currently, "any_input" is used to detect the both places.
    auto p_shape_of = ov::pass::pattern::wrap_type<ov::op::v3::ShapeOf>({ov::pass::pattern::any_input()});
    auto p_current_len = ov::pass::pattern::wrap_type<ov::op::v8::Gather>({p_shape_of, ov::pass::pattern::wrap_const(), ov::pass::pattern::wrap_const()});

    auto p_neg_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto p_neg_const_convert = ov::pass::pattern::optional<ov::op::v0::Convert>(p_neg_const);
    auto p_neg_const_reshape = ov::pass::pattern::optional<ov::op::v1::Reshape>({p_neg_const_convert, ov::pass::pattern::any_input()});
    auto p_neg_mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({p_current_len, p_neg_const_reshape});

    // For now, it has always been a constant, but this may change in the future.
    // In case of model being in FP16, there will be a decompressing subgraph:
    // i.e. Constant -> Convert -> Slice
    //
    // Also, it hasn't been observed yet, but, theoretically, there can also be a
    // dequantizing subgraph, so it's going to be any_input() here.
    auto p_rotary_emb_sincos = ov::pass::pattern::any_input();
    // the rotary_emb_cos/rotary_emb_sin are sliced by the total length [1,..4096,1,128]
    auto p_slice_1 =
        ov::pass::pattern::wrap_type<ov::op::v8::Slice>({p_rotary_emb_sincos, ov::pass::pattern::wrap_const(), p_opt_reshape, ov::pass::pattern::wrap_const(), ov::pass::pattern::wrap_const()});
    auto p_slice_2 = ov::pass::pattern::wrap_type<ov::op::v8::Slice>({p_slice_1, p_neg_mul, ov::pass::pattern::wrap_const(), ov::pass::pattern::wrap_const(), ov::pass::pattern::wrap_const()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto max_context_len = pattern_map.at(p_max_context_len).get_node_shared_ptr();
        if (max_context_len->get_friendly_name() != "max_context_len") {
            return false;
        }
        auto rotary_emb_sincos = pattern_map.at(p_rotary_emb_sincos).get_node_shared_ptr();
        auto slice_1 = pattern_map.at(p_slice_1).get_node_shared_ptr();
        auto slice_2 = pattern_map.at(p_slice_2).get_node_shared_ptr();

        auto axis = ov::op::v0::Constant::create(element::i64, Shape{}, {1});
        // in case of PagedAttention (Continuous batching) the rotary_emb_cos/rotary_emb_sin
        // are used not in the sequential order, so we need to use position_ids to get the expected values.
        auto gather = std::make_shared<ov::op::v8::Gather>(slice_1->input_value(0), position_ids, axis);
        gather->set_friendly_name(slice_2->get_friendly_name());
        gather->validate_and_infer_types();

        auto pshape = rotary_emb_sincos->get_output_partial_shape(0);
        if (pshape.rank().is_dynamic() || pshape.rank().get_length() != 4) {
            return false;
        }

        // PagedAttention expects the next layout for Q,K,V:
        // [batch_size_in_tokens, num_kv_heads * head_size]
        // so here we need to reshape the output tensor to move the seq dim (num tokens) to the batch
        // num_kv_heads * head_size are already handled in the StateManagementPattern transformation
        auto head_size = static_cast<int64_t>(pshape[3].get_length());
        auto new_shape = ov::op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{-1, 1, 1, head_size});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(gather, new_shape, false);
        replace_node(slice_2, reshape);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(p_slice_2, matcher_name);
    register_matcher(m, callback);
}

ov::pass::PositionIDsReplacerCodeGen2::PositionIDsReplacerCodeGen2(const std::shared_ptr<ov::op::v0::Parameter>& position_ids) {
    MATCHER_SCOPE(PositionIDsReplacerCodeGen2);

    auto p_range = ov::pass::pattern::wrap_type<ov::op::v4::Range>();
    auto p_power = ov::pass::pattern::wrap_type<ov::op::v1::Power>();
    auto p_einsum = ov::pass::pattern::wrap_type<ov::op::v7::Einsum>({p_range, p_power});
    auto p_sin_cos = ov::pass::pattern::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({p_einsum});
    auto p_reshape = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({p_sin_cos, ov::pass::pattern::any_input()});
    auto p_tile = ov::pass::pattern::wrap_type<ov::op::v0::Tile>({p_reshape, ov::pass::pattern::any_input()});
    auto p_opt_reshape = ov::pass::pattern::optional<ov::op::v1::Reshape>({p_tile, ov::pass::pattern::any_input()});
    auto p_opt_unsq = ov::pass::pattern::optional<ov::op::v0::Unsqueeze>({p_opt_reshape, ov::pass::pattern::any_input()});

    auto p_reshape_1in = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});
    auto p_add_2in = ov::pass::pattern::wrap_type<ov::op::v1::Add>({ov::pass::pattern::any_input(), ov::pass::pattern::any_input()});
    auto p_slice = ov::pass::pattern::wrap_type<ov::op::v8::Slice>({p_opt_unsq, p_reshape_1in, p_add_2in, ov::pass::pattern::wrap_const(), ov::pass::pattern::wrap_const()});

    auto p_add = ov::pass::pattern::wrap_type<ov::op::v1::Add>();
    matcher_pass_callback callback = [=, &position_ids](ov::pass::pattern::Matcher& m) {
        auto pvm = m.get_pattern_value_map();
        auto slice = pvm.at(p_slice).get_node_shared_ptr();

        auto gather = std::make_shared<ov::op::v8::Gather>(slice->input_value(0),
                                                   position_ids,
                                                   ov::op::v0::Constant::create(element::i64, Shape{}, {1}));
        if (gather->output(0).get_partial_shape().rank() != 3) {
            return false;
        }

        auto transpose =
            std::make_shared<ov::op::v1::Transpose>(gather, ov::op::v0::Constant::create(element::i64, Shape{3}, {1, 0, 2}));

        replace_node(slice, transpose);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(p_slice, matcher_name);
    register_matcher(m, callback);
}