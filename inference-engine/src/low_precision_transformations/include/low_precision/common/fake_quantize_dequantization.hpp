// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

typedef std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> FakeQuantizeDequantizationValues;

class FakeQuantizeDequantization {
public:
    FakeQuantizeDequantization();

    FakeQuantizeDequantization(
        const Output<Node>& data,
        const std::shared_ptr<ngraph::opset1::Convert>& convert,
        const std::shared_ptr<ngraph::opset1::Subtract>& subtract,
        const std::shared_ptr<ngraph::opset1::Convert>& subtractConvert,
        const std::shared_ptr<ngraph::opset1::Constant>& subtractConstant,
        const std::shared_ptr<ngraph::opset1::Multiply>& multiply,
        const std::shared_ptr<ngraph::opset1::Constant>& multiplyConstant);

    bool empty() const;
    bool multiplyHasZero() const;
    bool isShared() const;
    bool isLowPrecision() const;

    static bool checkElementwise(const std::shared_ptr<ngraph::Node>& elementwise);

    static void fillMultiply(
        const std::shared_ptr<ngraph::opset1::Multiply>& multiply,
        std::shared_ptr<ngraph::opset1::Constant>& constant);

    static void fillSubtract(
        const std::shared_ptr<ngraph::opset1::Subtract>& subtract,
        std::shared_ptr<ngraph::opset1::Convert>& convert,
        std::shared_ptr<ngraph::opset1::Constant>& constant);

    Output<Node> data;
    std::shared_ptr<opset1::Convert> convert;
    std::shared_ptr<opset1::Subtract> subtract;
    std::shared_ptr<ngraph::opset1::Convert> subtractConvert;
    std::shared_ptr<ngraph::opset1::Constant> subtractConstant;
    std::shared_ptr<opset1::Multiply> multiply;
    std::shared_ptr<ngraph::opset1::Constant> multiplyConstant;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
