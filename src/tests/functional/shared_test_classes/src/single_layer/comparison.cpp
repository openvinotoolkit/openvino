// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/single_layer/comparison.hpp"

using namespace LayerTestsDefinitions::ComparisonParams;
using namespace ngraph::helpers;

namespace LayerTestsDefinitions {
std::string ComparisonLayerTest::getTestCaseName(const testing::TestParamInfo<ComparisonTestParams> &obj) {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision ngInputsPrecision;
    ComparisonTypes comparisonOpType;
    InputLayerType secondInputType;
    InferenceEngine::Precision ieInPrecision;
    InferenceEngine::Precision ieOutPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes,
             ngInputsPrecision,
             comparisonOpType,
             secondInputType,
             ieInPrecision,
             ieOutPrecision,
             targetName,
             additional_config) = obj.param;
    std::ostringstream results;

    results << "IS0=" << ov::test::utils::vec2str(inputShapes.first) << "_";
    results << "IS1=" << ov::test::utils::vec2str(inputShapes.second) << "_";
    results << "inputsPRC=" << ngInputsPrecision.name() << "_";
    results << "comparisonOpType=" << comparisonOpType << "_";
    results << "secondInputType=" << secondInputType << "_";
    if (ieInPrecision != InferenceEngine::Precision::UNSPECIFIED) {
        results << "IEInPRC=" << ieInPrecision.name() << "_";
    }
    if (ieOutPrecision != InferenceEngine::Precision::UNSPECIFIED) {
        results << "IEOutPRC=" << ieOutPrecision.name() << "_";
    }
    results << "targetDevice=" << targetName;
    return results.str();
}

void ComparisonLayerTest::SetUp() {
    InputShapesTuple inputShapes;
    InferenceEngine::Precision ngInputsPrecision;
    InputLayerType secondInputType;
    InferenceEngine::Precision ieInPrecision;
    InferenceEngine::Precision ieOutPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes,
             ngInputsPrecision,
             comparisonOpType,
             secondInputType,
             ieInPrecision,
             ieOutPrecision,
             targetDevice,
             additional_config) = this->GetParam();

    auto ngInputsPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(ngInputsPrecision);
    configuration.insert(additional_config.begin(), additional_config.end());

    inPrc = ieInPrecision;
    outPrc = ieOutPrecision;

    ov::ParameterVector inputs {std::make_shared<ov::op::v0::Parameter>(ngInputsPrc, ov::Shape(inputShapes.first))};

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto secondInput = ngraph::builder::makeInputLayer(ngInputsPrc, secondInputType, inputShapes.second);
    OPENVINO_SUPPRESS_DEPRECATED_END
    if (secondInputType == InputLayerType::PARAMETER) {
        inputs.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(secondInput));
    }

    auto comparisonNode = ngraph::builder::makeComparison(inputs[0], secondInput, comparisonOpType);
    function = std::make_shared<ov::Model>(comparisonNode, inputs, "Comparison");
}

InferenceEngine::Blob::Ptr ComparisonLayerTest::GenerateInput(const InferenceEngine::InputInfo &inputInfo) const {
    InferenceEngine::Blob::Ptr blob;

    if (comparisonOpType == ComparisonTypes::IS_FINITE || comparisonOpType == ComparisonTypes::IS_NAN) {
        blob = make_blob_with_precision(inputInfo.getTensorDesc());
        blob->allocate();
        auto dataPtr = blob->buffer().as<float*>();
        auto dataPtrInt = blob->buffer().as<int*>();
        const auto range = blob->size();
        const float start = -static_cast<float>(range) / 2.f;
        testing::internal::Random random(1);

        for (size_t i = 0; i < range; i++) {
            if (i % 7 == 0) {
                dataPtr[i] = std::numeric_limits<float>::infinity();
            } else if (i % 7 == 1) {
                dataPtr[i] = -std::numeric_limits<float>::infinity();
            } else if (i % 7 == 2) {
                dataPtrInt[i] = 0x7F800000 + random.Generate(range);
            } else if (i % 7 == 3) {
                dataPtr[i] = std::numeric_limits<double>::quiet_NaN();
            } else if (i % 7 == 5) {
                dataPtr[i] = -std::numeric_limits<double>::quiet_NaN();
            } else {
                dataPtr[i] = start + static_cast<float>(random.Generate(range));
            }
        }
    } else {
        blob = LayerTestsUtils::LayerTestsCommon::GenerateInput(inputInfo);
    }

    return blob;
}

} // namespace LayerTestsDefinitions
