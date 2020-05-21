// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>

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

    auto fq = std::make_shared<opset1::FakeQuantize>(in, inputLowNode, inputHighNode, outputLowNode, outputHighNode, levels);

    return fq;
}

std::shared_ptr<ngraph::Node> makeFakeQuantize(const ngraph::Output<ngraph::Node> &in,
                                               const ngraph::element::Type &type,
                                               std::size_t levels,
                                               std::vector<size_t> constShapes) {
    size_t constDataSize = ngraph::shape_size(constShapes);
    std::vector<float> inputLowData, inputHighData, outputLowData, outputHighData;
    inputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize);
    if (levels != 2) {
        inputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize);
        outputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize);
        outputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize);
    } else {
        inputHighData = inputLowData;
        outputLowData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize);
        outputHighData = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(constDataSize);

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

    auto inputLowNode = ngraph::builder::makeConstant(type, constShapes, inputLowData, inputLowData.empty());
    auto inputHighNode = ngraph::builder::makeConstant(type, constShapes, inputHighData, inputHighData.empty());
    auto outputLowNode = ngraph::builder::makeConstant(type, constShapes, outputLowData, outputLowData.empty());
    auto outputHighNode = ngraph::builder::makeConstant(type, constShapes, outputHighData, outputHighData.empty());

    auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(in, inputLowNode, inputHighNode, outputLowNode, outputHighNode, levels);

    return fq;
}

}  // namespace builder
}  // namespace ngraph