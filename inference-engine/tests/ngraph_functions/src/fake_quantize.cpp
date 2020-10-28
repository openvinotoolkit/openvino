// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeFakeQuantize(const ngraph::Output<Node> &in,
                                       const element::Type &type,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes,
                                       const std::vector<float> &inputLowData,
                                       const std::vector<float> &inputHighData,
                                       const std::vector<float> &outputLowData,
                                       const std::vector<float> &outputHighData) {
    auto inputLowNode = makeConstant(type, constShapes, inputLowData, inputLowData.empty());
    auto inputHighNode = makeConstant(type, constShapes, inputHighData, inputHighData.empty());
    auto outputLowNode = makeConstant(type, constShapes, outputLowData, outputLowData.empty());
    auto outputHighNode = makeConstant(type, constShapes, outputHighData, outputHighData.empty());

    auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(in, inputLowNode, inputHighNode, outputLowNode, outputHighNode, levels);

    return fq;
}

std::shared_ptr<ngraph::Node> makeFakeQuantize(const ngraph::Output<ngraph::Node> &in,
                                               const ngraph::element::Type &type,
                                               std::size_t levels,
                                               std::vector<size_t> constShapes,
                                               const int32_t  seed) {
    size_t constDataSize = ngraph::shape_size(constShapes);
    std::vector<float> inputLowData, inputHighData, outputLowData, outputHighData;
    inputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
    if (levels != 2) {
        inputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
    } else {
        inputHighData = inputLowData;
        outputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize, 10, 1, seed);

        for (int i = 0; i < constDataSize; i++) {
            if (outputLowData[i] > outputHighData[i]) {
                outputLowData[i] = 1;
                outputHighData[i] = 0;
            } else {
                outputLowData[i] = 0;
                outputHighData[i] = 1;
            }
        }
    }

    for (int i = 0; i < constDataSize; i++) {
        inputLowData[i] = std::min(inputLowData[i], inputHighData[i]);
        inputHighData[i] = std::max(inputLowData[i], inputHighData[i]);
        if (inputLowData[i] == inputHighData[i])
            inputHighData[i] += 1;
    }

    for (int i = 0; i < constDataSize; i++) {
        outputLowData[i] = std::min(outputLowData[i], outputHighData[i]);
        outputHighData[i] = std::max(outputLowData[i], outputHighData[i]);
        if (outputLowData[i] == outputHighData[i])
            outputHighData[i] += 1;
    }

    auto inputLowNode = ngraph::builder::makeConstant(type, constShapes, inputLowData, inputLowData.empty(), seed);
    auto inputHighNode = ngraph::builder::makeConstant(type, constShapes, inputHighData, inputHighData.empty(), seed);
    auto outputLowNode = ngraph::builder::makeConstant(type, constShapes, outputLowData, outputLowData.empty(), seed);
    auto outputHighNode = ngraph::builder::makeConstant(type, constShapes, outputHighData, outputHighData.empty(), seed);

    auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(in, inputLowNode, inputHighNode, outputLowNode, outputHighNode, levels);

    return fq;
}

}  // namespace builder
}  // namespace ngraph
