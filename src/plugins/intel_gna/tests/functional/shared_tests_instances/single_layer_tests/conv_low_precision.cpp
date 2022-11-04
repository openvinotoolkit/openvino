// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "../../shared_tests_instances/skip_tests_check.hpp"
#include <gtest/gtest.h>


namespace ConvLowPrecicionTestNs {

    using namespace ngraph;
    using namespace ngraph::builder;
    using namespace ngraph::element;
    using namespace ngraph::op;
    using namespace ngraph::opset1;
    using namespace std;

    using ConvLowPrecisionTestParams = tuple<InferenceEngine::Precision, // Network Precision
                                             string,                     // Target Device
                                             map<string, string>,        // Configuration
                                             Shape,                      // Input Shape
                                             pair<float, float>,         // FQ Min and Max (before conv)
                                             std::size_t                 // Levels
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

        tie(netPrecision, targetDevice, configuration, inputShape, fqMinMax, levels) =
            obj.param;

        ostringstream result;
        result << "_netPRC=" << netPrecision.name();
        result << "_targetDevice=" << targetDevice;
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << ":" << configItem.second;
        }
        result << "_inputShape=" << CommonTestUtils::vec2str(inputShape);
        result << "_fqMinMax=(" << fqMinMax.first << ".." << fqMinMax.second << ")";
        result << "_levels=" << levels;

        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(),
                                                -0.05f,
                                                0.05f,
                                                1 / inputDataResolution);
    }

    ParameterVector createInputVector(const Type& type, const vector<std::size_t>& shapes) {
        return makeParams(type, {shapes});
    }

    shared_ptr<FakeQuantize> createFQNode(const Type& type,
                                          const shared_ptr<ov::Node>& node,
                                          float fqMin, float fqMax,
                                          std::size_t levels) {
        auto fqInpMin = makeConstant<float>(type, {1}, {fqMin});
        auto fqInpMax = makeConstant<float>(type, {1}, {fqMax});
        auto fqOutMin = makeConstant<float>(type, {1}, {fqMin});
        auto fqOutMax = makeConstant<float>(type, {1}, {fqMax});
        return make_shared<FakeQuantize>(node,
                                        fqInpMin,
                                        fqInpMax,
                                        fqOutMin,
                                        fqOutMax,
                                        levels);
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
        tie(netPrecision, targetDevice, configuration,
            inputShape, fqMinMax, levels) = this->GetParam();
        tie(fqMin, fqMax) = fqMinMax;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        size_t kernelHeight = inputShape[2] == 1 ? 1 : 2;

        // Create network
        auto inputVector = createInputVector(ngPrc, {inputShape});
        auto inputFQ = createFQNode(ngPrc, inputVector[0], fqMin, fqMax, levels);
        auto kernelWeights = makeConstant<float>(ngPrc, {8, 8, kernelHeight, 2}, {0.1f});
        auto weightsFQ = createFQNode(ngPrc, kernelWeights, fqMin, fqMax, levels);
        auto convolution = make_shared<Convolution>(inputFQ,
                                                    weightsFQ,
                                                    vector<std::size_t>{1, 1},
                                                    vector<ptrdiff_t>{0, 0},
                                                    vector<ptrdiff_t>{0, 0},
                                                    vector<std::size_t>{1, 1},
                                                    PadType::VALID);
        auto outputFQ = createFQNode(ngPrc, convolution, fqMin, fqMax, levels);

        //
        function = make_shared<ngraph::Function>(outputFQ, inputVector, "ConvLowPrecision");

        gnaVersionCheck.SetUp(targetDevice);

    }
};

using ConvLowPrecisionTestLib35 = ConvLowPrecisionTest;

TEST_P(ConvLowPrecisionTest, CompareWithRefs) {
    Run();
};

TEST_P(ConvLowPrecisionTestLib35, CompareWithRefs) {
   
    if (gnaVersionCheck.gnaLibVersionLessThan(3.5)) {
        GTEST_SKIP() << "Disabled test due to GNA library version is less than " << 3.5 << std::endl;
        return;
    }
    Run();
};

const vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32
};

const vector<map<string, string>> configs_3_0 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
};

const vector<map<string, string>> configs_3_5 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}},
};

const vector<map<string, string>> configs_2_0 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PRECISION", "I8"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_PRECISION", "I16"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}},
};

const Shape conv1D = {1, 8, 1, 16};
const Shape conv2D = {1, 8, 16, 16};

const vector<Shape> inputShapes = {
    conv1D,      // for convolution 1D
    conv2D       // for convolution 2D
};

const vector<pair<float, float>> fqMinMax = {
    {-8, 8}
};

const vector<std::size_t> levels = {
    numeric_limits<uint8_t>::max(),
    numeric_limits<uint16_t>::max()
};

INSTANTIATE_TEST_SUITE_P(smoke_LowPrecision20,
                         ConvLowPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_2_0),
                                            ::testing::Values(conv1D),
                                            ::testing::ValuesIn(fqMinMax),
                                            ::testing::ValuesIn(levels)),
                         ConvLowPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LowPrecision30,
                         ConvLowPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_3_0),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(fqMinMax),
                                            ::testing::ValuesIn(levels)),
                         ConvLowPrecisionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LowPrecision35,
                         ConvLowPrecisionTestLib35,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_3_5),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(fqMinMax),
                                            ::testing::ValuesIn(levels)),
                         ConvLowPrecisionTest::getTestCaseName);
}  // namespace ConvLowPrecicionTestNs
