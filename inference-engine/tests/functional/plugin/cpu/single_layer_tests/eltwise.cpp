// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/eltwise.hpp>
#include <ngraph_functions/builders.hpp>
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        LayerTestsDefinitions::EltwiseTestParams,
        CPUSpecificParams> EltwiseLayerCPUTestParamsSet;

class EltwiseLayerCPUTest : public testing::WithParamInterface<EltwiseLayerCPUTestParamsSet>,
                                     virtual public LayerTestsUtils::LayerTestsCommon, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj) {
        LayerTestsDefinitions::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;

        std::ostringstream result;
        result << LayerTestsDefinitions::EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::EltwiseTestParams>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void SetUp() {
        LayerTestsDefinitions::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();

        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::InputLayerType secondaryInputType;
        CommonTestUtils::OpType opType;
        ngraph::helpers::EltwiseTypes eltwiseType;
        std::map<std::string, std::string> additional_config;
        std::tie(inputShapes, eltwiseType, secondaryInputType, opType, netPrecision, inPrc, outPrc, inLayout, targetDevice, additional_config) = basicParamsSet;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        selectedType = getPrimitiveType() + "_" + netPrecision.name();

        std::vector<size_t> inputShape1, inputShape2;
        if (inputShapes.size() == 1) {
            inputShape1 = inputShape2 = inputShapes.front();
        } else if (inputShapes.size() == 2) {
            inputShape1 = inputShapes.front();
            inputShape2 = inputShapes.back();
        } else {
            IE_THROW() << "Incorrect number of input shapes";
        }

        configuration.insert(additional_config.begin(), additional_config.end());
        auto input = ngraph::builder::makeParams(ngPrc, {inputShape1});

        std::vector<size_t> shape_input_secondary;
        switch (opType) {
            case CommonTestUtils::OpType::SCALAR: {
                shape_input_secondary = std::vector<size_t>({1});
                break;
            }
            case CommonTestUtils::OpType::VECTOR:
                shape_input_secondary = inputShape2;
                break;
            default:
                FAIL() << "Unsupported Secondary operation type";
        }

        std::shared_ptr<ngraph::Node> secondaryInput;
        if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
            eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
            std::vector<float> data(ngraph::shape_size(shape_input_secondary));
            data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape_input_secondary), 10, 2);
            secondaryInput = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, data);
        } else if (eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD)  {
            int negative_data_size = ngraph::shape_size(shape_input_secondary) / 2;
            int positive_data_size = ngraph::shape_size(shape_input_secondary) - negative_data_size;
            std::vector<float> negative_data(negative_data_size);
            std::vector<float> data(positive_data_size);
            negative_data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(negative_data_size, -10, -2);
            data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(positive_data_size, 10, 2);
            data.insert(data.end(), negative_data.begin(), negative_data.end());
            secondaryInput = ngraph::builder::makeConstant(ngPrc, shape_input_secondary, data);
        } else {
            secondaryInput = ngraph::builder::makeInputLayer(ngPrc, secondaryInputType, shape_input_secondary);
            if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
                input.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
            }
        }

        auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);

        function = makeNgraphFunction(ngPrc, input, eltwise, "Eltwise");
    }
};

TEST_P(EltwiseLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
    CheckPluginRelatedResults(executableNetwork, "Eltwise");
}

namespace {

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinInp = {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesDiffInp = { // Different number of input nodes depending on optimizations
        ngraph::helpers::EltwiseTypes::POWER,
        // ngraph::helpers::EltwiseTypes::MOD // Does not execute because of transformations
};

std::map<std::string, std::string> additional_config;

std::vector<Precision> netPrc = {Precision::BF16, Precision::FP32};


std::vector<std::vector<std::vector<size_t>>> inShapes_4D = {
        {{2, 4, 4, 1}},
        {{2, 17, 5, 4}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c, nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
};

const auto params_4D = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_4D),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::ValuesIn(secondaryInputTypes),
            ::testing::ValuesIn(opTypes),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder, EltwiseLayerCPUTest, params_4D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netPrc),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_emptyCPUSpec, EltwiseLayerCPUTest, params_4D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<std::vector<size_t>>> inShapes_5D = {
        {{2, 4, 3, 4, 1}},
        {{2, 17, 7, 5, 4}},
        {{2, 17, 6, 5, 4}, {1, 17, 6, 1, 1}},
        {{2, 17, 6, 5, 1}, {1, 17, 1, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c, nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
};

const auto params_5D = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_5D),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::ValuesIn(secondaryInputTypes),
            ::testing::ValuesIn(opTypes),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder, EltwiseLayerCPUTest, params_5D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netPrc),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D, EltwiseLayerCPUTest, params_5D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<std::vector<size_t>>> inShapes_4D_Blocked_Planar = {
        {{2, 17, 31, 3}, {2, 1, 31, 3}},
        {{2, 17, 5, 1}, {2, 1, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_4D_Blocked_Planar = {
        CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {}),
};

const auto params_4D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_4D_Blocked_Planar),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(opTypes),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Planar)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Planar, EltwiseLayerCPUTest, params_4D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_4D_Planar_Blocked = {
        {{2, 1, 31, 3}, {2, 17, 31, 3}},
        {{2, 1, 1, 4}, {2, 17, 5, 1}},
};

std::vector<CPUSpecificParams> cpuParams_4D_Planar_Blocked = {
        CPUSpecificParams({nchw, nChw16c}, {nChw16c}, {}, {}),
};

const auto params_4D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_4D_Planar_Blocked),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(opTypes),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar_Blocked)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Planar_Blocked, EltwiseLayerCPUTest, params_4D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_5D_Blocked_Planar = {
        {{2, 17, 31, 4, 3}, {2, 1, 31, 1, 3}},
        {{2, 17, 5, 3, 1}, {2, 1, 1, 3, 4}},
};

std::vector<CPUSpecificParams> cpuParams_5D_Blocked_Planar = {
        CPUSpecificParams({nCdhw16c, ncdhw}, {nCdhw16c}, {}, {}),
};

const auto params_5D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
            ::testing::ValuesIn(inShapes_5D_Blocked_Planar),
            ::testing::ValuesIn(eltwiseOpTypesBinInp),
            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
            ::testing::ValuesIn(opTypes),
            ::testing::ValuesIn(netPrc),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Precision::FP32),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Planar)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Blocked_Planar, EltwiseLayerCPUTest, params_5D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_5D_Planar_Blocked = {
        {{2, 1, 31, 1, 3}, {2, 17, 31, 4, 3}},
        {{2, 1, 1, 3, 4}, {2, 17, 5, 3, 1}},
};

std::vector<CPUSpecificParams> cpuParams_5D_Planar_Blocked = {
        CPUSpecificParams({ncdhw, nCdhw16c}, {nCdhw16c}, {}, {}),
};

const auto params_5D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D_Planar_Blocked),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netPrc),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Planar_Blocked)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Planar_Blocked, EltwiseLayerCPUTest, params_5D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<std::vector<size_t>>> inShapes_4D_1D = {
        {{2, 17, 5, 4}, {4}},
        {{1, 3, 3, 3}, {3}},
};

std::vector<CPUSpecificParams> cpuParams_4D_1D = {
        CPUSpecificParams({nChw16c, x}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc, x}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, x}, {nchw}, {}, {})
};

const auto params_4D_1D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_1D),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netPrc),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D, EltwiseLayerCPUTest, params_4D_1D, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<std::vector<size_t>>> inShapes_5D_1D = {
        {{2, 17, 5, 4, 10}, {10}},
        {{1, 3, 3, 3, 3}, {3}},
};

std::vector<CPUSpecificParams> cpuParams_5D_1D = {
        CPUSpecificParams({nCdhw16c, x}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc, x}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, x}, {ncdhw}, {}, {})
};

const auto params_5D_1D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_5D_1D),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netPrc),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Precision::FP32),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D)));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D, EltwiseLayerCPUTest, params_5D_1D, EltwiseLayerCPUTest::getTestCaseName);


} // namespace
} // namespace CPULayerTestsDefinitions
