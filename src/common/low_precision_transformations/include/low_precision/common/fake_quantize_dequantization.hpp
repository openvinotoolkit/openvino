// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <low_precision/lpt_visibility.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

typedef std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> FakeQuantizeDequantizationValues;

class LP_TRANSFORMATIONS_API FakeQuantizeDequantization {
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

    bool empty() const noexcept;
    bool multiplyHasZeroOrDenormal() const;
    bool isShared() const;
    bool isLowPrecision() const;
    std::shared_ptr<Node> copyWithNewInput(const std::shared_ptr<Node>& input) const;
    void scalarizeConstants();

    static bool checkElementwise(const std::shared_ptr<ngraph::Node>& elementwise);

    static bool checkShape(const std::shared_ptr<ngraph::Node>& elementwise);

    static int fillDequantizationParams(
        const std::shared_ptr<ngraph::Node>& elementwise,
        std::shared_ptr<ngraph::opset1::Convert>& convert,
        std::shared_ptr<ngraph::opset1::Constant>& constant);

    static int fillDequantizationParams(
        const std::shared_ptr<ngraph::Node>& elementwise,
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
