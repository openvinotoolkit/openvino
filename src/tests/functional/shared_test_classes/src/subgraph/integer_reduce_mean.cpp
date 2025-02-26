// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/integer_reduce_mean.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace test {

std::string IntegerReduceMeanTest::getTestCaseName(const testing::TestParamInfo<IntegerReduceMeanParams>& obj) {
    ov::element::Type input_precision;
    std::vector<size_t> input_shape;
    std::vector<size_t> axes;
    bool quantized;
    const char *device;
    std::tie(input_precision, input_shape, axes, quantized, device) = obj.param;
    std::ostringstream result;
    result << "inputPrecision=" << input_precision.to_string() << "_";
    result << "inputShape=" << ov::test::utils::vec2str(input_shape) << "_";
    result << "axes=" << ov::test::utils::vec2str(axes) << "_";
    result << "device=" + std::string(device);
    if (quantized)
        result << "quantized=true";
    else
        result << "quantized=false";
    return result.str();
}

void IntegerReduceMeanTest::SetUp() {
    ov::element::Type input_precision;
    std::vector<size_t> input_shape;
    std::vector<size_t> axes;
    std::vector<size_t> axes_shape;
    bool quantized;
    std::tie(input_precision, input_shape, axes, quantized, targetDevice) = this->GetParam();
    axes_shape.push_back(axes.size());

    auto dataNode = std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape(input_shape));
    auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape(axes_shape), axes);

    std::shared_ptr<ov::op::v1::ReduceMean> reduce_mean;
    if (quantized) {
        std::vector<size_t> dataFqConstShapes(input_shape.size(), 1);
        size_t constDataSize = ov::shape_size(dataFqConstShapes);
        std::vector<float> inputLowData(constDataSize), inputHighData(constDataSize), outputLowData(constDataSize), outputHighData(constDataSize);
        for (size_t i = 0; i < constDataSize; i++) {
            inputLowData[i] = 0;
            inputHighData[i] = 255;
            outputLowData[i] = 0;
            outputHighData[i] = 255;
        }
        auto dataFqNode = ov::test::utils::make_fake_quantize(
            dataNode, input_precision, 256, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);
        reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(dataFqNode, axesNode, true);
    } else {
        reduce_mean = std::make_shared<ov::op::v1::ReduceMean>(dataNode, axesNode, true);
    }

    ov::ParameterVector inputs;
    inputs.push_back(dataNode);
    ov::ResultVector outputs;
    outputs.push_back(std::make_shared<ov::op::v0::Result>(reduce_mean));
    function = std::make_shared<ov::Model>(outputs, inputs);
}

}  // namespace test
}  // namespace ov
