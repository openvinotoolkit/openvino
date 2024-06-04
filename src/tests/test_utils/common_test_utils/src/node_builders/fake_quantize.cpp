// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/fake_quantize.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/fake_quantize.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& in,
                                             const ov::element::Type& type,
                                             std::size_t levels,
                                             std::vector<size_t> constShapes,
                                             const std::vector<float>& inputLowData,
                                             const std::vector<float>& inputHighData,
                                             const std::vector<float>& outputLowData,
                                             const std::vector<float>& outputHighData) {
    auto inputLowNode = make_constant(type, constShapes, inputLowData);
    auto inputHighNode = make_constant(type, constShapes, inputHighData);
    auto outputLowNode = make_constant(type, constShapes, outputLowData);
    auto outputHighNode = make_constant(type, constShapes, outputHighData);

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(in,
                                                         inputLowNode,
                                                         inputHighNode,
                                                         outputLowNode,
                                                         outputHighNode,
                                                         levels);
    return fq;
}

std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& in,
                                             const ov::element::Type& type,
                                             std::size_t levels,
                                             std::vector<size_t> constShapes) {
    size_t constDataSize = ov::shape_size(constShapes);
    std::vector<float> inputLowData, inputHighData, outputLowData, outputHighData;
    inputLowData = ov::test::utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1);
    if (levels != 2) {
        inputHighData = ov::test::utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1);
        outputLowData = ov::test::utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1);
        outputHighData = ov::test::utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1);
    } else {
        inputHighData = inputLowData;
        outputLowData = ov::test::utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1);
        outputHighData = ov::test::utils::generateVector<ov::element::Type_t::f32>(constDataSize, 10, 1);

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

    auto inputLowNode = ov::test::utils::make_constant(type, constShapes, inputLowData);
    auto inputHighNode = ov::test::utils::make_constant(type, constShapes, inputHighData);
    auto outputLowNode = ov::test::utils::make_constant(type, constShapes, outputLowData);
    auto outputHighNode = ov::test::utils::make_constant(type, constShapes, outputHighData);

    auto fq = std::make_shared<ov::op::v0::FakeQuantize>(in,
                                                         inputLowNode,
                                                         inputHighNode,
                                                         outputLowNode,
                                                         outputHighNode,
                                                         levels);

    return fq;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
