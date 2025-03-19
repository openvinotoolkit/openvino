// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include "low_precision/lpt_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace pass {
namespace low_precision {

typedef std::tuple<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> FakeQuantizeDequantizationValues;

class LP_TRANSFORMATIONS_API FakeQuantizeDequantization {
public:
    FakeQuantizeDequantization();

    FakeQuantizeDequantization(
        const Output<Node>& data,
        const std::shared_ptr<ov::opset1::Convert>& convert,
        const std::shared_ptr<ov::opset1::Subtract>& subtract,
        const std::shared_ptr<ov::opset1::Convert>& subtractConvert,
        const std::shared_ptr<ov::opset1::Constant>& subtractConstant,
        const std::shared_ptr<ov::opset1::Multiply>& multiply,
        const std::shared_ptr<ov::opset1::Constant>& multiplyConstant);

    bool empty() const noexcept;
    bool multiplyHasZeroOrDenormal() const;
    bool isShared() const;
    bool isLowPrecision() const;
    bool isPerTensor() const;
    ov::element::Type getPrecision() const;
    std::shared_ptr<Node> copyWithNewInput(const std::shared_ptr<Node>& input) const;

    bool checkElementwise(const std::shared_ptr<ov::Node>& elementwise) const;

    static bool checkShape(const std::shared_ptr<ov::Node>& elementwise);

    static int fillDequantizationParams(
        const std::shared_ptr<ov::Node>& elementwise,
        std::shared_ptr<ov::opset1::Convert>& convert,
        std::shared_ptr<ov::opset1::Constant>& constant);

    static int fillDequantizationParams(
        const std::shared_ptr<ov::Node>& elementwise,
        std::shared_ptr<ov::opset1::Constant>& constant);

    // for most node with layout NC, NCHW, NCDWH, index of channel dimension is 1
    size_t channelDimIndex = 1ul;
    Output<Node> data;
    std::shared_ptr<ov::opset1::Convert> convert;
    std::shared_ptr<ov::opset1::Subtract> subtract;
    std::shared_ptr<ov::opset1::Convert> subtractConvert;
    std::shared_ptr<ov::opset1::Constant> subtractConstant;
    std::shared_ptr<ov::opset1::Multiply> multiply;
    std::shared_ptr<ov::opset1::Constant> multiplyConstant;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
