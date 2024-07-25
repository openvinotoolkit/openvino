// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_stft(const NodeContext& context) {
    // stft(input: Tensor, n_fft: int, hop_length: Optional[int] = None,
    //      win_length: Optional[int] = None, window: Optional[Tensor] = None,
    //      center: bool = True, pad_mode: str = 'reflect', normalized: bool = False,
    //      onesided: Optional[bool] = None,
    //      return_complex: Optional[bool] = None) -> Tensor

    num_inputs_check(context, 2, 10);

    auto input = context.get_input(0);
    auto n_fft = context.get_input(1);
    auto hop_length = context.get_input(2);
    // auto hop_length = context.const_input<int64_t>(2);
    // auto win_length = context.get_input(3);
    auto window = context.get_input(4);
    // auto center = context.get_input(5);
    // auto pad_mode = context.get_input(6);
    // auto normalized = context.get_input(7);
    // auto onesided = context.get_input(8);
    // auto return_complex = context.get_input(9);

    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));

    // Torch stft accept input of [signal] or [bs, signal], convert always to [bs, signal] for OV.
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto class_probs_shape = context.mark_node(std::make_shared<v8::Gather>(input_shape, const_neg_1, const_0));
    auto inp_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{const_neg_1, class_probs_shape}, 0));
    input = context.mark_node(std::make_shared<v1::Reshape>(input, inp_shape, false));

    constexpr bool transpose_frames = true;
    auto stft = context.mark_node(std::make_shared<v15::STFT>(input, window, n_fft, hop_length, transpose_frames));

    return {stft};
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
