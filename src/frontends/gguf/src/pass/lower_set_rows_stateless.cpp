// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass/lower_set_rows_stateless.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/gguf/set_rows_op.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
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

        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        std::shared_ptr<ov::Node> res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, indices, data, axes);

        // Multi-sequence: if the destination is a Reshape, reshape the scatter result back to the
        // original [1, n_seq, ctx_per_seq, emb] layout (ctx_per_seq stays dynamic for llama-bench).
        if (auto dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
            auto dst_ps = dst_reshape->get_input_partial_shape(0);
            std::vector<int64_t> shape = {dst_ps[0].get_length(), dst_ps[1].get_length(),
                                          dst_ps[2].is_static() ? dst_ps[2].get_length() : -1,
                                          dst_ps[3].get_length()};
            res = std::make_shared<ov::op::v1::Reshape>(
                res, ov::op::v0::Constant::create(ov::element::i64, {4}, shape), false);
        }

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
