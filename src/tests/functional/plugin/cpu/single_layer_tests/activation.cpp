// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/activation.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {

using ActivationLayerCPUTestParamSet = std::tuple<
        std::vector<InputShape>,                                         // Input shapes
        std::vector<size_t>,                                             // Activation shapes
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>>, // Activation type and constant value
        InferenceEngine::Precision,                                      // Net precision
        InferenceEngine::Precision,                                      // Input precision
        InferenceEngine::Precision,                                      // Output precision
        CPUSpecificParams>;

class ActivationLayerCPUTest : public testing::WithParamInterface<ActivationLayerCPUTestParamSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    ActivationTypes activationType;
    static std::string getTestCaseName(const testing::TestParamInfo<ActivationLayerCPUTestParamSet> &obj) {
        std::vector<InputShape> inputShapes;
        std::vector<size_t> activationShapes;
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationTypeAndConstValue;
        Precision netPrecision, inPrecision, outPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::activationNames[activationTypeAndConstValue.first] << "_";
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto &shape : inputShapes) {
                result << CommonTestUtils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << "AS=" << CommonTestUtils::vec2str(activationShapes) << "_";
        result << "ConstantsValue=" << CommonTestUtils::vec2str(activationTypeAndConstValue.second) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrecision.name() << "_";
        result << "outPRC=" << outPrecision.name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        int32_t startFrom;
        uint32_t range;
        int32_t resolution;
        if (activationType == ActivationTypes::Exp && netPrecision == Precision::BF16) {
            startFrom = 0;
            range = 2;
            resolution = 32768;
        } else if (activationType == ActivationTypes::Acosh) {
            startFrom = 2;
            range = 2;
            resolution = 128;
        } else {
            startFrom = 0;
            range = 15;
            resolution = 32768;
        }
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (funcInput.get_element_type().is_real()) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i],
                                                                 range, startFrom, resolution);
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::vector<size_t> activationShapes;
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationTypeAndConstValue;
        Precision inPrecision, outPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, activationShapes, activationTypeAndConstValue, netPrecision, inPrecision, outPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        activationType = activationTypeAndConstValue.first;
        auto constantsValue = activationTypeAndConstValue.second;

        inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrecision);
        outType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(outPrecision);
        selectedType = getPrimitiveType() + "_" + netPrecision.name();

        init_input_shapes(inputShapes);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeDynamicParams(ngPrc, {inputDynamicShapes.front()});
        auto activation = ngraph::builder::makeActivation(params[0], ngPrc, activationType, activationShapes, constantsValue);
        activation->get_rt_info() = getCPUInfo();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{activation}, params, "Activation");
    }

    InferenceEngine::Precision netPrecision;
};

TEST_P(ActivationLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
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
        {GeluTanh,    {{}}},
        {SoftSign,    {{}}}
};

std::vector<Precision> netPrc = {
        Precision::BF16,
        Precision::FP32
};

/* ============= Activation (1D) ============= */
std::vector<CPUSpecificParams> cpuParams_3D = {
        CPUSpecificParams({nCw16c}, {nCw16c}, {}, {}),
        CPUSpecificParams({nwc}, {nwc}, {}, {}),
        CPUSpecificParams({ncw}, {ncw}, {}, {})
};

std::vector<std::vector<ov::Shape>> basic3D = {
        {{2, 4, 4}},
        {{2, 17, 5}},
};

const auto basicCases3D = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(basic3D)),
        ::testing::Values(activationShapes),
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrc),
        ::testing::Values(Precision::FP32),
        ::testing::Values(Precision::FP32),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_3D))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases3D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (2D) ============= */
std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw}, {nchw}, {}, {})
};

std::vector<std::vector<ov::Shape>> basic4D = {
            {{2, 4, 4, 1}},
            {{2, 17, 5, 4}}
};


const auto basicCases4D = ::testing::Combine(
            ::testing::ValuesIn(static_shapes_to_test_representation(basic4D)),
            ::testing::Values(activationShapes),
            ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(Precision::FP32),
            ::testing::Values(Precision::FP32),
            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Eltwise_CPU_BF16, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (3D) ============= */
std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})
};

std::vector<std::vector<ov::Shape>> basic5D = {
        {{2, 4, 3, 4, 1}},
        {{2, 17, 7, 5, 4}}
};

const auto basicCases5D = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(basic5D)),
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
        InferenceEngine::Precision::FP32
};

std::vector<CPUSpecificParams> cpuParamsDynamicMath = {
        CPUSpecificParams({}, {}, {}, {})
};

std::vector<std::vector<InputShape>> dynamicMathBasic = {
        {
                {{{-1, -1}, {{1, 50}, {5, 128}, {3, 64}}}},
                {{{-1, -1, -1, -1, -1, -1, -1, -1}, {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 3, 2, 3, 2, 3, 2, 3}, {3, 3, 3, 3, 3, 3, 3, 3}}}},
                {{{{1, 5}, 128}, {{1, 128}, {3, 128}, {5, 128}}}}
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
