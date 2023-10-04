// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeFakeQuantize(const ov::Output<Node>& in,
                                       const element::Type& constantType,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes,
                                       const std::vector<float>& inputLowData,
                                       const std::vector<float>& inputHighData,
                                       const std::vector<float>& outputLowData,
                                       const std::vector<float>& outputHighData) {
    auto inputLowNode = makeConstant(constantType, constShapes, inputLowData, inputLowData.empty());
    auto inputHighNode = makeConstant(constantType, constShapes, inputHighData, inputHighData.empty());
    auto outputLowNode = makeConstant(constantType, constShapes, outputLowData, outputLowData.empty());
    auto outputHighNode = makeConstant(constantType, constShapes, outputHighData, outputHighData.empty());

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(in,
                                                         inputLowNode,
                                                         inputHighNode,
                                                         outputLowNode,
                                                         outputHighNode,
                                                         levels);

    return fq;
}

std::shared_ptr<ov::Node> makeFakeQuantize(const ov::Output<ov::Node>& in,
                                           const ov::element::Type& type,
                                           std::size_t levels,
                                           std::vector<size_t> constShapes,
                                           const int32_t seed) {
    size_t constDataSize = ov::shape_size(constShapes);
    std::vector<float> inputLowData, inputHighData, outputLowData, outputHighData;
    inputLowData = NGraphFunctions::Utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1, seed);
    if (levels != 2) {
        inputHighData = NGraphFunctions::Utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputLowData = NGraphFunctions::Utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputHighData = NGraphFunctions::Utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1, seed);
    } else {
        inputHighData = inputLowData;
        outputLowData = NGraphFunctions::Utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1, seed);
        outputHighData = NGraphFunctions::Utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1, seed);

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

    auto inputLowNode =
        ngraph::builder::makeConstant(type, constShapes, inputLowData, inputLowData.empty(), 10.f, 1.f, seed);
    auto inputHighNode =
        ngraph::builder::makeConstant(type, constShapes, inputHighData, inputHighData.empty(), 10.f, 1.f, seed);
    auto outputLowNode =
        ngraph::builder::makeConstant(type, constShapes, outputLowData, outputLowData.empty(), 10.f, 1.f, seed);
    auto outputHighNode =
        ngraph::builder::makeConstant(type, constShapes, outputHighData, outputHighData.empty(), 10.f, 1.f, seed);

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(in,
                                                         inputLowNode,
                                                         inputHighNode,
                                                         outputLowNode,
                                                         outputHighNode,
                                                         levels);

    return fq;
}

}  // namespace builder
}  // namespace ngraph
