// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include "openvino/frontend/complex_type_mark.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_stft(const NodeContext& context) {
    // schema: aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool
    // normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
    //
    // Note: aten::stft doesn't have "center" and "pad_mode" attrs like torch.stft, so the number of the inputs is lower
    // and index of any input after the "window" is smaller accordingly

    num_inputs_check(context, 2, 8);

    auto input = context.get_input(0);
    auto n_fft = context.get_input(1);

    ov::Output<ov::Node> hop_length;
    if (!context.input_is_none(2)) {
        hop_length = context.get_input(2);
    } else {
        // Defualt floor(n_fft / 4)
        const auto four = context.mark_node(std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, 4));
        const auto four_cast = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(four, n_fft));
        hop_length = context.mark_node(std::make_shared<ov::op::v1::Divide>(n_fft, four_cast));
    }

    ov::Output<ov::Node> win_length;
    if (!context.input_is_none(3)) {
        win_length = context.get_input(3);
    } else {
        win_length = n_fft;
    }

    ov::Output<ov::Node> window;
    if (!context.input_is_none(4)) {
        window = context.get_input(4);
    } else {
        const auto one = context.mark_node(std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{}, 1));
        const auto one_cast = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(one, input));
        const auto zero = context.mark_node(std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{1}, 0));
        const auto win_length_cast =
            context.mark_node(std::make_shared<ov::op::v0::Convert>(win_length, ov::element::i64));
        const auto win_len_vec = context.mark_node(std::make_shared<ov::op::v0::Unsqueeze>(win_length_cast, zero));
        window = context.mark_node(std::make_shared<ov::op::v3::Broadcast>(one_cast, win_len_vec));
    }

    bool normalized = false;
    if (!context.input_is_none(5)) {
        normalized = context.const_input<bool>(5);
    }

    bool onesided = true;
    if (!context.input_is_none(6)) {
        onesided = context.const_input<bool>(6);
    }
    PYTORCH_OP_CONVERSION_CHECK(onesided, "aten::stft conversion is currently supported with onesided=True only.");

    bool return_complex = false;
    if (!context.input_is_none(7)) {
        return_complex = context.const_input<bool>(7);
    }

    // Perform STFT
    constexpr bool transpose_frames = true;
    auto stft = context.mark_node(std::make_shared<v15::STFT>(input, window, n_fft, hop_length, transpose_frames));

    if (normalized) {
        const auto nfft_convert = context.mark_node(std::make_shared<v1::ConvertLike>(n_fft, stft));
        const auto divisor = context.mark_node(std::make_shared<v0::Sqrt>(nfft_convert));
        stft = context.mark_node(std::make_shared<v1::Divide>(stft, divisor));
    }
    if (return_complex) {
        return {context.mark_node(std::make_shared<ComplexTypeMark>(stft, stft->get_element_type()))};
    } else {
        return {stft};
    }
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
