// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/activation.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;

namespace CPULayerTestsDefinitions  {

using inputShapesPair = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;

using ActivationLayerCPUTestParamSet = std::tuple<
        inputShapesPair,                                                 // Input shapes
        std::vector<size_t>,                                             // Activation shapes
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>>, // Activation type and constant value
        InferenceEngine::Precision,                                      // Net precision
        InferenceEngine::Precision,                                      // Input precision
        InferenceEngine::Precision,                                      // Output precision
        CPUSpecificParams>;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                            virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj) {
        inputShapesPair inputShapes;
        std::vector<size_t> activationShapes;
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationTypeAndConstValue;
        Precision netPrecision, inPrecision, outPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::activationNames[activationTypeAndConstValue.first] << "_";
        if (!inputShapes.first.empty()) {
            result << "IS=" << CommonTestUtils::partialShape2str(inputShapes.first) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes.second) {
            result << "(";
            for (const auto & item : shape) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
            result << ")_";
        }
        result << "AS=" << CommonTestUtils::vec2str(activationShapes) << "_";
        result << "ConstantsValue=" << CommonTestUtils::vec2str(activationTypeAndConstValue.second) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrecision.name() << "_";
        result << "outPRC=" << outPrecision.name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override {
        int32_t data_start_from;
        uint32_t data_range;
        int32_t resolution;

        if (activationType == ActivationTypes::Exp && netPrecision == Precision::BF16) {
            data_start_from = 0;
            data_range = 2;
            resolution = 32768;
        } else {
            data_start_from = 0;
            data_range = 15;
            resolution = 32768;
        }

        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), data_range, data_start_from, resolution);
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        inputShapesPair inputShapes;
        std::vector<size_t> activationShapes;
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationTypeAndConstValue;
        Precision inPrecision, outPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        activationType = activationTypeAndConstValue.first;
        auto constantsValue = activationTypeAndConstValue.second;

        selectedType = getPrimitiveType() + "_" + netPrecision.name();

        targetStaticShapes.reserve(inputShapes.second.size());
        for (size_t i = 0; i < inputShapes.second.size(); i++) {
            targetStaticShapes.push_back(inputShapes.second[i]);
        }
        inputDynamicShapes = inputShapes.first;

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShapes.front().front()});
        auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, activationShapes, constantsValue);
        activation->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params, "Activation");
    }

    InferenceEngine::Precision netPrecision;
};

TEST_P(ActivationLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Eltwise");
}


namespace {
// list only types supported by eltwise
const std::vector<size_t> activationShapes = {};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sqrt,        {{}}},
        {Sigmoid,     {{}}},
        {Tanh,        {{}}},
        {Relu,        {{}}},
        {Exp,         {{}}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Elu,         {{0.1f}}},
        {Swish,       {{0.1f}}},
        {HSwish,      {{}}},
        {Mish,        {{}}},
        {PReLu,       {{-0.01f}}},
        {GeluErf,     {{}}},
        {GeluTanh,    {{}}}
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

const std::vector<inputShapesPair> basic4D = {
        {
                {},
                // Static shape
                {
                        {{2, 4, 4, 1}}
                }
        },
        {
                {},
                // Static shape
                {
                        {{2, 17, 5, 4}}
                }
        }
};

std::vector<Precision> netPrc = {Precision::BF16, Precision::FP32};

const auto basicCases4D = ::testing::Combine(
            ::testing::ValuesIn(basic4D),
            ::testing::Values(activationShapes),
            ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::FP32),
            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

const std::vector<inputShapesPair> basic5D = {
        {
                {},
                // Static shape
                {
                        {{2, 4, 3, 4, 1}}
                }
        },
        {
                {},
                // Static shape
                {
                        {{2, 17, 7, 5, 4}}
                }
        }
};

const auto basicCases5D = ::testing::Combine(
        ::testing::ValuesIn(basic5D),
        ::testing::Values(activationShapes),
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(Precision::FP32),
        ::testing::Values(Precision::FP32),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesDynamicMath = {
        {Log,         {{}}},
        {Sign,        {{}}},
        {Acos,        {{}}},
        {Acosh,       {{}}},
        {Asin,        {{}}},
        {Asinh,       {{}}},
        {Atan,        {{}}},
        {Atanh,       {{}}},
        {Cos,         {{}}},
        {Cosh,        {{}}},
        {Tan,         {{}}},
        {HardSigmoid, {{0.2f, 0.5f}}},
        {Selu,        {{1.6732f, 1.0507f}}},
        {Ceiling,     {{}}}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

std::vector<CPUSpecificParams> cpuParamsDynamicMath = {
        CPUSpecificParams({}, {}, {}, {})
};

const std::vector<inputShapesPair> dynamicMathBasic = {
        {
                // dynamic
                {
                        {-1, -1}
                },
                // target
                {
                        {{1, 50}},
                        {{5, 128}},
                        {{3, 64}}
                }
        },
        {
                // dynamic
                {
                        {-1, -1, -1, -1, -1, -1, -1, -1}
                },
                // target
                {
                        {{2, 2, 2, 2, 2, 2, 2, 2}},
                        {{2, 3, 2, 3, 2, 3, 2, 3}},
                        {{3, 3, 3, 3, 3, 3, 3, 3}}
                }
        },
        {
                // dynamic
                {
                        {{1, 5}, 128}
                },
                // target
                {
                        {{1, 128}},
                        {{3, 128}},
                        {{5, 128}}
                }
        }
};

const auto dynamicMathBasicCases = ::testing::Combine(
        ::testing::ValuesIn(dynamicMathBasic),
        ::testing::Values(activationShapes),
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesDynamicMath)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(Precision::FP32),
        ::testing::Values(Precision::FP32),
        ::testing::ValuesIn(cpuParamsDynamicMath)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_dynamicMath_CPU, ActivationLayerCPUTest, dynamicMathBasicCases, ActivationLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
