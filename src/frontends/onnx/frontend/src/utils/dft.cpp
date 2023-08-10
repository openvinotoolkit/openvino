// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft.hpp"

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/core/deprecated.hpp"

using namespace ngraph::onnx_import;

namespace ngraph {
namespace onnx_import {
namespace dft {

namespace {
// For DFT, IDFT, IRDFT cases, if real signal are provided (with shape [D_0, D_1, ..., D_{N-1}, 1])
// it's needed to fill tensors with zero imaginary part to be aligned with Core ops requirements.
bool try_convert_real_to_complex(ov::Output<ov::Node>& signal) {
    if (signal.get_partial_shape().rank().is_static()) {
        const auto length = signal.get_partial_shape().rank().get_length();
        const auto last_axis_pos = length - 1;
        const auto last_dim = signal.get_partial_shape()[last_axis_pos];
        if (last_dim.is_static() && last_dim.get_length() == 1) {
            ov::Output<ov::Node> imag_part = default_opset::Constant::create(signal.get_element_type(), {}, {0});
            imag_part =
                std::make_shared<default_opset::Broadcast>(imag_part, std::make_shared<default_opset::ShapeOf>(signal));
            signal = std::make_shared<default_opset::Concat>(OutputVector{signal, imag_part}, last_axis_pos);
            return true;
        }
    }
    // [D_0, D_1, ..., D_{N-1}, 2] case, so additional transformations not needed or we are not able to check it during
    // importing.
    return false;
}
}  // namespace

ov::Output<ov::Node> make_dft(const ov::Output<ov::Node>& signal,
                              const ov::Output<ov::Node>& length,
                              int64_t axis,
                              bool is_inversed,
                              bool is_onesided) {
    auto processed_signal = signal;
    const auto axis_const = default_opset::Constant::create(element::i64, {1}, {axis});
    bool conversion_to_complex_applied = false;
    if (is_inversed || !is_onesided) {  // skip for RDFT case
        conversion_to_complex_applied = try_convert_real_to_complex(processed_signal);
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool dft_length_provided = !ngraph::op::is_null(length);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ov::Output<ov::Node> result;
    if (is_inversed) {
        if (is_onesided) {
            result = dft_length_provided ? std::make_shared<default_opset::IRDFT>(processed_signal, axis_const, length)
                                         : std::make_shared<default_opset::IRDFT>(processed_signal, axis_const);
            if (conversion_to_complex_applied) {  // align the output shape with a real numbers representation
                const auto unsqueeze_axis = default_opset::Constant::create(element::i64, {}, {-1});
                result = std::make_shared<default_opset::Unsqueeze>(result, unsqueeze_axis);
            }
        } else {
            result = dft_length_provided ? std::make_shared<default_opset::IDFT>(processed_signal, axis_const, length)
                                         : std::make_shared<default_opset::IDFT>(processed_signal, axis_const);
        }
    } else {
        if (is_onesided) {
            result = dft_length_provided ? std::make_shared<default_opset::RDFT>(processed_signal, axis_const, length)
                                         : std::make_shared<default_opset::RDFT>(processed_signal, axis_const);
        } else {
            result = dft_length_provided ? std::make_shared<default_opset::DFT>(processed_signal, axis_const, length)
                                         : std::make_shared<default_opset::DFT>(processed_signal, axis_const);
        }
    }
    return {result};
}
}  // namespace  dft
}  // namespace onnx_import
}  // namespace ngraph
