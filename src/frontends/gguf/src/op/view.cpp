// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "utils.hpp"
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
namespace ov {
namespace frontend {
namespace gguf {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (context.get_attribute<int>("op_case", 0) == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix({process_view_input(context, 0, dst_shape[2] * dst_shape[3])},
                                          context.get_name());
    }
    if (context.get_attribute<int>("op_case", 0) == 3) {
        auto input = context.get_input(0);
        auto input_ov_shape = input.get_partial_shape();
        auto input_ggml_shape = context.get_attribute<ov::Shape>("input_ggml_shape");

        // Input already reshaped: restore the original ggml shape before slicing.
        if (input_ov_shape.size() != input_ggml_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, {input_ggml_shape.size()}, input_ggml_shape), false);
        }

        auto dst_shape = context.get_output_shape().to_shape();

        // find the index of dst_shape that is different from input shape, and use that index to slice the input
        int slice_dim = -1;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] != input_ggml_shape[i]) {
                slice_dim = i;
                break;
            }
        }
        // Identity view: nothing differs, so nothing to slice (avoids dst_shape[-1]).
        if (slice_dim < 0) {
            return {input};
        }

        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {dst_shape[slice_dim]});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim});
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
        return {sliced};
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
