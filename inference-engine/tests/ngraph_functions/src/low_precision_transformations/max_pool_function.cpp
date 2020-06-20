// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MaxPoolFunction::getOriginal(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, inputShape);

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(input, ngPrecision, 256ul, { 1ul });

    const auto shapeReshapeBefore = ngraph::opset1::Constant::create(
        ngraph::element::i64,
        ngraph::Shape{ 6ul },
        ngraph::Shape{ inputShape[0], inputShape[1] / 4ul, 2ul, 2ul, inputShape[2], inputShape[3] });
    const auto reshapeBefore = std::make_shared<ngraph::opset1::Reshape>(fakeQuantize, shapeReshapeBefore, false);
    reshapeBefore->set_friendly_name("reshapeBefore");

    const auto permutation = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 1, 4, 2, 5, 3 });
    const auto permute = std::make_shared<ngraph::opset1::Transpose>(reshapeBefore, permutation);
    permute->set_friendly_name("permute");

    const auto shapeReshapeAfter = ngraph::opset1::Constant::create(
        ngraph::element::i64,
        ngraph::Shape{ 4 },
        ngraph::Shape{ 1, inputShape[1] / 4ul, inputShape[2] * 2, inputShape[3] * 2 });
    const auto reshapeAfter = std::make_shared<ngraph::opset1::Reshape>(permute, shapeReshapeAfter, false);
    reshapeAfter->set_friendly_name("reshapeAfter");

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshapeAfter }, ngraph::ParameterVector{ input });
    return function;
}

std::shared_ptr<ngraph::Function> MaxPoolFunction::getReference(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape) {
    return nullptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
