// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hyper_connection_mix.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_HC_MIX (glm5next mHC): inputs mixes [..., (2+n)*n], scale [3], base [(2+n)*n].
// Output [..., (2+n)*n] = concat(pre[n], post[n], comb[n*n] as [src][dst]), so the builder can VIEW the
// three parts like it does with llama.cpp's mixes. Lowers to the fused ov::op::internal::HyperConnectionMix.
OutputVector translate_hc_mix(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    const auto n = static_cast<int64_t>(context.get_attribute<int>("n_streams"));
    const auto iters = static_cast<int64_t>(context.get_attribute<int>("sinkhorn_iterations"));
    const auto eps = context.get_attribute<float>("eps");
    auto mix = std::make_shared<ov::op::internal::HyperConnectionMix>(context.get_input(0),
                                                                      context.get_input(1),
                                                                      context.get_input(2),
                                                                      n,
                                                                      iters,
                                                                      eps);
    // comb [..., n, n] -> [..., n*n]
    auto in_shape = std::make_shared<ov::op::v3::ShapeOf>(context.get_input(0), ov::element::i64);
    auto lead = std::make_shared<ov::op::v8::Slice>(in_shape,
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
    auto comb_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{lead, ov::op::v0::Constant::create(ov::element::i64, {1}, {n * n})},
        0);
    auto comb = std::make_shared<ov::op::v1::Reshape>(mix->output(2), comb_shape, false);
    auto out = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{mix->output(0), mix->output(1), comb}, -1);
    return rename_outputs_with_suffix({out}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
