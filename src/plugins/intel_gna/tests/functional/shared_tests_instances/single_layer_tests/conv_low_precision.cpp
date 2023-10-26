// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../../shared_tests_instances/skip_tests_check.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "openvino/opsets/opset12.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace ConvLowPrecicionTestNs {

using namespace ov;
using namespace ngraph::builder;
using namespace ov::element;
using namespace ov::op;
using namespace ov::opset12;
using namespace std;

using ConvLowPrecisionTestParams = tuple<InferenceEngine::Precision,  // Network Precision
                                         string,                      // Target Device
                                         map<string, string>,         // Configuration
                                         Shape,                       // Input Shape
                                         pair<float, float>,          // FQ Min and Max (before conv)
                                         std::size_t                  // Levels
                                         >;

class ConvLowPrecisionTest : public testing::WithParamInterface<ConvLowPrecisionTestParams>,
                             public LayerTestsUtils::LayerTestsCommon {
    float fqMin = 0.0f;
    float fqMax = 0.0f;
    float inputDataResolution = 1.0f;

public:
    static string getTestCaseName(testing::TestParamInfo<ConvLowPrecisionTestParams> obj) {
        InferenceEngine::Precision netPrecision;
        string targetDevice;
        map<string, string> configuration;
        Shape inputShape;
        pair<float, float> fqMinMax;
        std::size_t levels = 0;

        tie(netPrecision, targetDevice, configuration, inputShape, fqMinMax, levels) = obj.param;

        ostringstream result;
        result << "_netPRC=" << netPrecision.name();
        result << "_targetDevice=" << targetDevice;
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << ":" << configItem.second;
        }
        result << "_inputShape=" << ov::test::utils::vec2str(inputShape);
        result << "_fqMinMax=(" << fqMinMax.first << ".." << fqMinMax.second << ")";
        result << "_levels=" << levels;

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlobFloatNormalDistribution(info.getTensorDesc(), 0.0f, 0.2f, 7235346);
    }

    ParameterVector createInputVector(const Type& type, const vector<std::size_t>& shapes) {
        return ov::ParameterVector{std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(shapes))};
    }

    shared_ptr<FakeQuantize> createFQNode(const Type& type,
                                          const shared_ptr<ov::Node>& node,
                                          float fqMin,
                                          float fqMax,
                                          std::size_t levels) {
        auto fqInpMin = makeConstant<float>(type, {1}, {fqMin});
        auto fqInpMax = makeConstant<float>(type, {1}, {fqMax});
        auto fqOutMin = makeConstant<float>(type, {1}, {fqMin});
        auto fqOutMax = makeConstant<float>(type, {1}, {fqMax});
        return make_shared<FakeQuantize>(node, fqInpMin, fqInpMax, fqOutMin, fqOutMax, levels);
    }

protected:
    GnaLayerTestCheck gnaVersionCheck;

    void SetUp() override {
        // Loosen threshold because of precision decrease during test
        threshold = 0.1;

        // Receive test params
        InferenceEngine::Precision netPrecision;
        vector<std::size_t> inputShape;
        pair<float, float> fqMinMax;
        std::size_t levels = 0;
        tie(netPrecision, targetDevice, configuration, inputShape, fqMinMax, levels) = this->GetParam();
        tie(fqMin, fqMax) = fqMinMax;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        size_t kernelHeight = inputShape[2] == 1 ? 1 : 2;

        // Create network
        auto inputVector = createInputVector(ngPrc, {inputShape});
        auto inputFQ = createFQNode(ngPrc, inputVector[0], fqMin, fqMax, levels);
        auto kernelWeights = makeConstant<float>(ngPrc, {8, 8, kernelHeight, 2}, {}, true, 1.0f, -1.0f, 7235346);
        auto weightsFQ = createFQNode(ngPrc, kernelWeights, fqMin, fqMax, levels);
        auto convolution = make_shared<Convolution>(inputFQ,
                                                    weightsFQ,
                                                    vector<std::size_t>{1, 1},
                                                    vector<ptrdiff_t>{0, 0},
                                                    vector<ptrdiff_t>{0, 0},
                                                    vector<std::size_t>{1, 1},
                                                    PadType::VALID);
        auto outputFQ = createFQNode(ngPrc, convolution, fqMin, fqMax, levels);
        function = make_shared<ngraph::Function>(outputFQ, inputVector, "ConvLowPrecision");
        gnaVersionCheck.SetUp(targetDevice);
    }
};

TEST_P(ConvLowPrecisionTest, CompareWithRefs) {
    Run();
};

const vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                          InferenceEngine::Precision::FP32};

const vector<map<string, string>> configs_1 = {
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_1_0"}},
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_1_0"}},
};

const vector<map<string, string>> configs_2 = {
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
};

const vector<map<string, string>> configs_3 = {
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
    {{"GNA_DEVICE_MODE", "GNA_AUTO"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
};

const Shape conv1D = {1, 8, 1, 16};
const Shape conv2D = {1, 8, 16, 16};

const vector<Shape> inputShapes = {conv1D, conv2D};

const vector<pair<float, float>> fqMinMax = {{-1.0f, 1.0f}};

const vector<std::size_t> levels = {numeric_limits<uint8_t>::max(), numeric_limits<uint16_t>::max()};

INSTANTIATE_TEST_SUITE_P(smoke_LowPrecision10,
                         ConvLowPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_1),
                                            ::testing::Values(conv1D),
                                            ::testing::ValuesIn(fqMinMax),
                                            ::testing::ValuesIn(levels)),
                         ConvLowPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LowPrecision20,
                         ConvLowPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_2),
                                            ::testing::Values(conv1D),
                                            ::testing::ValuesIn(fqMinMax),
                                            ::testing::ValuesIn(levels)),
                         ConvLowPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LowPrecision3X,
                         ConvLowPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_3),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(fqMinMax),
                                            ::testing::ValuesIn(levels)),
                         ConvLowPrecisionTest::getTestCaseName);

}  // namespace ConvLowPrecicionTestNs
