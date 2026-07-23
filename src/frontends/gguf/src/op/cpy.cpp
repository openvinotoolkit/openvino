// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

#include <cstdint>
#include <memory>
#include <vector>
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    ov::Output<ov::Node> res =
        std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_attribute<ov::element::Type>("output_type"));

    // A CPY may reinterpret the source layout into its destination's (e.g. qwen3-next's conv-state
    // writeback flattens the contiguous [S, F] conv_state_last into the flat [S*F] recurrent cache
    // row). When the destination (output) shape differs from the source but holds the same number of
    // elements, reshape to the output layout so the model result matches the ggml cache tensor.
    const auto & out_ps = context.get_output_shape();
    const auto & in_ps = res.get_partial_shape();
    if (out_ps.is_static() && in_ps.is_static() && in_ps != out_ps) {
        const auto in_shape = in_ps.to_shape();
        const auto out_shape = out_ps.to_shape();
        int64_t in_elems = 1, out_elems = 1;
        for (auto d : in_shape)
            in_elems *= static_cast<int64_t>(d);
        for (auto d : out_shape)
            out_elems *= static_cast<int64_t>(d);
        // Only reinterpret when the element counts match. A zero-element CPY (empty defrag/no-op
        // writeback, e.g. a decode step's [.,.,0,.] conv-state copy) keeps its own shape -- reshaping
        // it to the non-empty cache layout would be an invalid element-count change.
        if (in_elems == out_elems && in_elems != 0) {
            std::vector<int64_t> tgt(out_shape.begin(), out_shape.end());
            res = std::make_shared<ov::op::v1::Reshape>(
                res, ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt), false);
        }
    } else if (out_ps.is_static() && in_ps.rank().is_static() && in_ps != out_ps) {
        // The input carries a (possibly spurious) dynamic axis but the destination cache row is fully
        // static (qwen3-next conv-state writeback: in [1,1,4096,1..3] -> cache [1,1,1,12288]). Reshape
        // to the static cache layout directly: at runtime the dynamic axis takes the value that makes
        // the element counts match, so a plain static target is valid. Guard on the static-dim product
        // dividing the output (a zero-element or incompatible copy keeps its own shape).
        int64_t in_static = 1;
        bool has_dyn = false;
        for (int64_t i = 0; i < in_ps.rank().get_length(); ++i) {
            if (in_ps[i].is_static()) {
                in_static *= in_ps[i].get_length();
            } else {
                has_dyn = true;
            }
        }
        const auto out_shape = out_ps.to_shape();
        int64_t out_elems = 1;
        for (auto d : out_shape)
            out_elems *= static_cast<int64_t>(d);
        if (has_dyn && in_static != 0 && out_elems % in_static == 0) {
            std::vector<int64_t> tgt(out_shape.begin(), out_shape.end());
            res = std::make_shared<ov::op::v1::Reshape>(
                res, ov::op::v0::Constant::create(ov::element::i64, {tgt.size()}, tgt), false);
        }
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
