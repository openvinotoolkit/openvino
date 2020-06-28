// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/check.hpp>
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

class FakeQuantizeDequantization {
public:
    FakeQuantizeDequantization();

    FakeQuantizeDequantization(
        ngraph::element::Type dataNodeOutputPrecision,
        ngraph::Shape dataNodeOutputShape,
        std::shared_ptr<ngraph::opset1::Convert> convert,
        std::shared_ptr<ngraph::opset1::Subtract> subtract,
        std::shared_ptr<ngraph::opset1::Multiply> multiply);

    ngraph::element::Type dataNodeOutputPrecision;
    ngraph::Shape dataNodeOutputShape;
    std::shared_ptr<ngraph::opset1::Convert> convert;
    std::shared_ptr<ngraph::opset1::Subtract> subtract;
    std::shared_ptr<ngraph::opset1::Multiply> multiply;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
