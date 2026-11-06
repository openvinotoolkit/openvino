// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/lower_set_rows_stateless.hpp"

#include <map>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::pass {

LowerSetRowsStateless::LowerSetRowsStateless() {
    auto set_rows_pattern = ov::pass::pattern::wrap_type<SetRows>();

    const auto callback = [](ov::pass::pattern::Matcher& m) {
        auto set_rows = ov::as_type_ptr<SetRows>(m.get_match_root());
        if (!set_rows) {
            return false;
        }
        auto data = set_rows->input_value(0);     // reshaped to [1, 1, seq, emb]
        auto indices = set_rows->input_value(1);  // squeezed row indices
        auto dst = set_rows->input_value(2);      // destination tensor (Parameter for a KV cache)

        // SET_ROWS indices address the flattened context/head row axis. Flatten the destination
        // the same way translate_set_rows flattened the updates; with a one-token cache the two
        // layouts happen to coincide, which previously hid this mismatch until the second step.
        const auto dst_shape = dst.get_partial_shape();
        const auto flatten = ov::as_type_ptr<ov::op::v1::Reshape>(data.get_node_shared_ptr());
        const auto original_shape = flatten ? flatten->input_value(0).get_partial_shape() : ov::PartialShape::dynamic();
        const auto original_rank = original_shape.rank();
        const bool has_static_row = original_rank.is_static() && original_rank.get_length() > 0 &&
                                    original_shape[original_rank.get_length() - 1].is_static();
        const int64_t row_width = has_static_row ? original_shape[original_rank.get_length() - 1].get_length() : 0;
        const bool full_token_rows = is_kvcache_4d_shape(dst_shape) && dst_shape[2].get_length() > 0 &&
                                     dst_shape[3].get_length() > 0 && has_static_row &&
                                     row_width % dst_shape[3].get_length() == 0 &&
                                     row_width / dst_shape[3].get_length() == dst_shape[2].get_length();
        if (full_token_rows) {
            // Cache [1, ctx, n_heads_kv, head_size], one token per row: write whole tokens on axis 1. This is the
            // pattern of the GPU plugin's StatelessKVFusion (read back with VariadicSplit, see translate_kv_view),
            // which writes the new rows in place instead of copying the whole cache first. Indices are per token.
            const int64_t n_heads_kv = dst_shape[2].get_length();
            const int64_t head_size = dst_shape[3].get_length();
            // translate_set_rows flattened the rows to [.., 1, tokens * n_heads_kv, head_size]; regroup them from the
            // rows it started with, in one reshape.
            data = flatten->input_value(0);
            auto tokens = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, n_heads_kv, head_size}),
                false);
            // A rank-1 axis: the form the plugin's StatelessKVFusion matches.
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
            std::shared_ptr<ov::Node> scattered =
                std::make_shared<ov::op::v3::ScatterUpdate>(dst, indices, tokens, axis);
            scattered->set_friendly_name(set_rows->get_friendly_name());
            ov::copy_runtime_info(set_rows, scattered);
            ov::replace_node(set_rows, scattered);
            return true;
        }
        OPENVINO_ASSERT(dst_shape.rank().is_static() && dst_shape[dst_shape.rank().get_length() - 1].is_static(),
                        "SET_ROWS requires a static destination row size");
        const auto row_size = dst_shape[dst_shape.rank().get_length() - 1].get_length();
        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, -1, row_size});
        auto flat_dst = std::make_shared<ov::op::v1::Reshape>(dst, flat_shape, true);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        std::shared_ptr<ov::Node> res = std::make_shared<ov::op::v3::ScatterUpdate>(flat_dst, indices, data, axes);

        ov::Output<ov::Node> output_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);

        // Keep the destination's logical layout. Storage aliases are rebound by TranslateSession.
        res = std::make_shared<ov::op::v1::Reshape>(res, output_shape, false);

        res->set_friendly_name(set_rows->get_friendly_name());
        ov::copy_runtime_info(set_rows, res);
        ov::replace_node(set_rows, res);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(set_rows_pattern, "gguf::LowerSetRowsStateless"),
                     callback);
}

namespace {

// The token read emitted by translate_kv_view: VariadicSplit(cache, 1, Concat(length, -1)) with the tail unused.
std::shared_ptr<ov::op::v0::Concat> kv_read_lengths(const std::shared_ptr<ov::op::v1::VariadicSplit>& split) {
    const auto axis = ov::as_type_ptr<ov::op::v0::Constant>(split->get_input_node_shared_ptr(1));
    const auto lengths = ov::as_type_ptr<ov::op::v0::Concat>(split->get_input_node_shared_ptr(2));
    if (!is_kvcache_4d_shape(split->get_input_partial_shape(0)) || !axis || shape_size(axis->get_shape()) != 1 ||
        axis->cast_vector<int64_t>()[0] != 1 || !lengths || lengths->get_input_size() != 2 ||
        split->get_output_size() != 2 || !split->output(1).get_target_inputs().empty()) {
        return nullptr;
    }
    const auto tail = ov::as_type_ptr<ov::op::v0::Constant>(lengths->get_input_node_shared_ptr(1));
    if (!tail || tail->cast_vector<int64_t>() != std::vector<int64_t>{-1}) {
        return nullptr;
    }
    return lengths;
}

}  // namespace

bool SliceSharedKVReads::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::map<ov::Output<ov::Node>, std::vector<std::shared_ptr<ov::op::v1::VariadicSplit>>> reads;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto split = ov::as_type_ptr<ov::op::v1::VariadicSplit>(node); split && kv_read_lengths(split)) {
            reads[split->input_value(0)].push_back(split);
        }
    }
    bool changed = false;
    for (const auto& [cache, splits] : reads) {
        if (splits.size() < 2) {
            continue;
        }
        for (const auto& split : splits) {
            const auto i64 = [](std::vector<int64_t> v) {
                return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{v.size()}, v);
            };
            const auto& shape = cache.get_partial_shape();
            const int64_t heads = shape[2].get_length(), head_size = shape[3].get_length();
            // Slicing the [1, ctx, heads * head_size] rows, not the 4D cache, is what keeps SDPA on its optimized
            // kernel.
            auto rows = std::make_shared<ov::op::v1::Reshape>(cache, i64({1, -1, heads * head_size}), false);
            auto slice = std::make_shared<ov::op::v8::Slice>(rows,
                                                             i64({0}),
                                                             kv_read_lengths(split)->input_value(0),
                                                             i64({1}),
                                                             i64({1}));
            auto tokens = std::make_shared<ov::op::v1::Reshape>(slice, i64({1, -1, heads, head_size}), false);
            tokens->set_friendly_name(split->get_friendly_name());
            ov::copy_runtime_info(split, {rows, slice, tokens});
            split->output(0).replace(tokens->output(0));
            changed = true;
        }
    }
    return changed;
}

}  // namespace ov::frontend::gguf::pass
