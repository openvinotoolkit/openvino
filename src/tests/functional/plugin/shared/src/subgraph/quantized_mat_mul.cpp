// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_mat_mul.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace test {

std::string QuantMatMulTest::getTestCaseName(const testing::TestParamInfo<QuantMatMulLayerTestParamsSet> &obj) {
    const auto& [quantParams0, quantParams1, element_type, inputShape0, inputShape1, targetDevice] = obj.param;

    const auto& [quantLevels0, inputRange0, outputRange0, quantGranularity0, fqPrec0] = quantParams0;
    const auto& [quantLevels1, inputRange1, outputRange1, quantGranularity1, fqPrec1] = quantParams1;
    std::ostringstream result;
    result << "IS0=" << ov::test::utils::vec2str(inputShape0) << "_";
    result << "IS1=" << ov::test::utils::vec2str(inputShape1) << "_";
    result << "Levels0=" << quantLevels0 << "_";
    result << "Levels1=" << quantLevels1 << "_";
    result << "inputRange0=" << inputRange0.first << "_" << inputRange0.second << "_";
    result << "outputRange0=" << outputRange0.first << "_" << outputRange0.second << "_";
    result << "inputRange1=" << inputRange1.first << "_" << inputRange1.second << "_";
    result << "outputRange1=" << outputRange1.first << "_" << outputRange1.second << "_";
    result << "QuantGranularity0=" << quantGranularity0 << "_";
    result << "QuantGranularity1=" << quantGranularity1 << "_";
    result << "fq0PRC=" << fqPrec0.get_type_name() << "_";
    result << "fq1PRC=" << fqPrec1.get_type_name() << "_";
    result << "ET=" << element_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantMatMulTest::SetUp() {
    const auto& [quantParams0, quantParams1, element_type, inputShape0, inputShape1, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    const auto& [quantLevels0, inputRange0, outputRange0, quantGranularity0, fqPrec0] = quantParams0;
    const auto& [quantLevels1, inputRange1, outputRange1, quantGranularity1, fqPrec1] = quantParams1;
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShape0)),
                                std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShape1))};

    auto makeFakeQuantizeNode = [element_type = element_type](size_t quantLevels,
                                                              QuantRange inputRange,
                                                              QuantRange outputRange,
                                                              ov::test::utils::QuantizationGranularity quantGranularity,
                                                              const ov::Output<ov::Node>& in,
                                                              ov::Shape inputShape,
                                                              ov::element::Type prec) -> std::shared_ptr<ov::Node> {
        std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
        if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
            dataFqConstShapes[1] = inputShape[1];
        size_t constDataSize = ov::shape_size(dataFqConstShapes);
        std::vector<float> inputLowData(constDataSize), inputHighData(constDataSize), outputLowData(constDataSize), outputHighData(constDataSize);
        for (int i = 0; i < constDataSize; i++) {
            inputLowData[i] = inputRange.first;
            inputHighData[i] = inputRange.second;
            outputLowData[i] = outputRange.first;
            outputHighData[i] = outputRange.second;
        }
        return ov::test::utils::make_fake_quantize(
            in, element_type, quantLevels, dataFqConstShapes, inputLowData, inputHighData, outputLowData, outputHighData);
    };

    auto dataFq0 = makeFakeQuantizeNode(quantLevels0, inputRange0, outputRange0, quantGranularity0, params[0], inputShape0, fqPrec0);
    auto dataFq1 = makeFakeQuantizeNode(quantLevels1, inputRange1, outputRange1, quantGranularity1, params[1], inputShape1, fqPrec1);

    auto MatMul = std::make_shared<ov::op::v0::MatMul>(dataFq0, dataFq1);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(MatMul)};
    function = std::make_shared<ov::Model>(results, params, "QuantMatMul");
}
}  // namespace test
}  // namespace ov
