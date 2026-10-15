// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/lower_set_rows_stateless.hpp"

#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

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
        if (is_kvcache_4d_shape(dst_shape)) {
            // Cache [1, ctx, n_heads_kv, head_size], one token per row: write whole tokens on axis 1. This is the
            // pattern of the GPU plugin's StatelessKVFusion (read back with VariadicSplit, see translate_kv_view),
            // which writes the new rows in place instead of copying the whole cache first. Indices are per token.
            const int64_t n_heads_kv = dst_shape[2].get_length();
            const int64_t head_size = dst_shape[3].get_length();
            // translate_set_rows flattened the rows to [.., 1, tokens * n_heads_kv, head_size]; regroup them from the
            // rows it started with, in one reshape.
            if (const auto flatten = ov::as_type_ptr<ov::op::v1::Reshape>(data.get_node_shared_ptr())) {
                data = flatten->input_value(0);
            }
            auto tokens = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, -1, n_heads_kv, head_size}),
                false);
            // A rank-1 axis: the form the plugin's StatelessKVFusion matches.
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
            std::shared_ptr<ov::Node> scattered = std::make_shared<ov::op::v3::ScatterUpdate>(dst, indices, tokens, axis);
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

}  // namespace ov::frontend::gguf::pass
