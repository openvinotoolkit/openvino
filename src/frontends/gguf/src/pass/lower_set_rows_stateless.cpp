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
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace pass {

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
        OPENVINO_ASSERT(dst_shape.rank().is_static() && dst_shape[dst_shape.rank().get_length() - 1].is_static(),
                        "SET_ROWS requires a static destination row size");
        const auto row_size = dst_shape[dst_shape.rank().get_length() - 1].get_length();
        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 1, -1, row_size});
        auto flat_dst = std::make_shared<ov::op::v1::Reshape>(dst, flat_shape, true);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        std::shared_ptr<ov::Node> res = std::make_shared<ov::op::v3::ScatterUpdate>(flat_dst, indices, data, axes);

        ov::Output<ov::Node> output_shape = std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64);

        // Multi-sequence: if the destination is a Reshape, reshape the scatter result back to the
        // original [1, n_seq, ctx_per_seq, emb] layout (ctx_per_seq stays dynamic for llama-bench).
        if (auto dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
            output_shape = std::make_shared<ov::op::v3::ShapeOf>(dst_reshape->input_value(0), ov::element::i64);
        }
        res = std::make_shared<ov::op::v1::Reshape>(res, output_shape, false);

        res->set_friendly_name(set_rows->get_friendly_name());
        ov::copy_runtime_info(set_rows, res);
        ov::replace_node(set_rows, res);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(set_rows_pattern, "gguf::LowerSetRowsStateless"),
                     callback);
}

}  // namespace pass
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
