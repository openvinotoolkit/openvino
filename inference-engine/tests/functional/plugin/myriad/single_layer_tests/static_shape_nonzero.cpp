// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/operations/static_shape_nonzero.hpp"

#include "vpu/private_plugin_config.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <precision_utils.h>
#include <ngraph/opsets/opset3.hpp>

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <random>

typedef std::tuple<
        InferenceEngine::SizeVector,    // Input shape
        InferenceEngine::Precision,     // Input precision
        LayerTestsUtils::TargetDevice   // Device name
> staticShapeNonZeroLayerTestParams;

namespace LayerTestsDefinitions {

class StaticShapeNonZeroLayerTest : public testing::WithParamInterface<staticShapeNonZeroLayerTestParams>,
                                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<staticShapeNonZeroLayerTestParams> obj) {
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision inputPrecision;
        std::string targetDevice;
        std::tie(inputShape, inputPrecision, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "inPrc=" << inputPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
        configuration[InferenceEngine::MYRIAD_DISABLE_REORDER] = CONFIG_VALUE(YES);

        InferenceEngine::SizeVector inputShape;
        std::tie(inputShape, inPrc, targetDevice) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);

        const auto input = std::make_shared<ngraph::opset3::Parameter>(ngPrc, ngraph::Shape(inputShape));
        const auto staticShapeNonZero = std::make_shared<ngraph::vpu::op::StaticShapeNonZero>(input, ngraph::element::i32);
        ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(staticShapeNonZero->output(0)),
                std::make_shared<ngraph::opset3::Result>(staticShapeNonZero->output(1))};
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{input});
        outPrc = InferenceEngine::Precision::I32;
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        return FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 4, -2, 2);
    }

    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutput,
                 const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override {
        const auto expectedIndicesPtr = reinterpret_cast<const int32_t*>(expectedOutput[0].second.data());
        const auto expectedDimsPtr = reinterpret_cast<const int32_t*>(expectedOutput[1].second.data());

        const auto actualIndices = actualOutputs[0];
        const auto actualDims = actualOutputs[1];

        const auto actualIndicesPtr = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualIndices)->rmap().as<const int32_t*>();
        const auto actualDimsPtr = InferenceEngine::as<InferenceEngine::MemoryBlob>(actualDims)->rmap().as<const int32_t*>();

        ASSERT_EQ(expectedDimsPtr[0], actualDimsPtr[0]);
        ASSERT_EQ(expectedDimsPtr[1], actualDimsPtr[1]);

        const auto totalDimsSize = actualIndices->getTensorDesc().getDims()[1];

        for (int axis = 0; axis < actualDimsPtr[0]; ++axis) {
            for (int i = 0; i < actualDimsPtr[1]; ++i) {
                const auto idx = i + axis * totalDimsSize;
                ASSERT_EQ(expectedIndicesPtr[idx], actualIndicesPtr[idx]);
            }
        }
    }
};

TEST_P(StaticShapeNonZeroLayerTest, accuracy) {
    Run();
}

std::vector<InferenceEngine::SizeVector> inputDims = {
        { 7 },
        { 1000 },
        { 3, 5 },
        { 65, 33 },
        { 33, 65 },
        { 1, 1000 },
        { 223, 217, 21 },
        { 3, 4, 5, 1 },
        { 3, 4, 1, 5, 1 }
};

std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_CASE_P(smoke_accuracy, StaticShapeNonZeroLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inputDims),
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace LayerTestsDefinitions
