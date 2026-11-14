// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/rdft.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft {
namespace {
// Shared handling of the Rfft/Irfft attributes. Both operators transform the last `signal_ndim` dimensions
// of the signal, which is expressed as the axes [-signal_ndim, ..., -1]; only one-sided, unnormalized
// transforms are defined by the operators.
std::shared_ptr<v0::Constant> fft_axes(const ov::frontend::onnx::Node& node) {
    const auto signal_ndim = node.get_attribute_value<int64_t>("signal_ndim", 1);
    CHECK_VALID_NODE(node, signal_ndim >= 1, "Attribute 'signal_ndim' must be positive. Got: ", signal_ndim);
    CHECK_VALID_NODE(node,
                     node.get_attribute_value<int64_t>("normalized", 0) == 0,
                     "Only 'normalized' = 0 is supported, as defined by the operator.");
    CHECK_VALID_NODE(node,
                     node.get_attribute_value<int64_t>("onesided", 1) == 1,
                     "Only 'onesided' = 1 is supported, as defined by the operator.");

    std::vector<int64_t> axes(static_cast<size_t>(signal_ndim));
    for (int64_t i = 0; i < signal_ndim; ++i) {
        axes[static_cast<size_t>(i)] = i - signal_ndim;
    }
    return v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
}

void check_fft_input_type(const ov::frontend::onnx::Node& node, const ov::Output<ov::Node>& x) {
    const auto& x_type = x.get_element_type();
    CHECK_VALID_NODE(
        node,
        x_type.is_dynamic() || x_type == ov::element::f16 || x_type == ov::element::f32 || x_type == ov::element::f64,
        "Unsupported input data type for X, expected FP16, FP32 or FP64 but got: ",
        x_type);
}
}  // namespace

namespace opset_1 {
ov::OutputVector rfft(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftrfft
    // One-sided real-to-complex FFT over the last signal_ndim dimensions of a real tensor [..., n]; the output
    // is [..., n/2 + 1, 2] with the real and imaginary parts in the last dimension, which is what RDFT returns.
    common::default_op_checks(node, 1, 1);
    const auto& x = node.get_ov_inputs()[0];
    check_fft_input_type(node, x);

    return {std::make_shared<v9::RDFT>(x, fft_axes(node))};
}

ov::OutputVector irfft(const ov::frontend::onnx::Node& node) {
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftirfft
    // Inverse of Rfft: the input [..., n/2 + 1, 2] holds the real and imaginary parts in the last dimension
    // and the output is the real signal [..., n]. IRDFT restores the even signal length n = 2 * (n/2 + 1 - 1).
    common::default_op_checks(node, 1, 1);
    const auto& x = node.get_ov_inputs()[0];
    check_fft_input_type(node, x);

    return {std::make_shared<v9::IRDFT>(x, fft_axes(node))};
}

ONNX_OP("Rfft", OPSET_SINCE(1), com_microsoft::opset_1::rfft, MICROSOFT_DOMAIN);
ONNX_OP("Irfft", OPSET_SINCE(1), com_microsoft::opset_1::irfft, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace ov::frontend::onnx::com_microsoft
