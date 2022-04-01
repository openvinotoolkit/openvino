// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/eltwise.hpp>
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        subgraph::EltwiseTestParams,
        CPUSpecificParams,
        fusingSpecificParams> EltwiseLayerCPUTestParamsSet;

class EltwiseLayerCPUTest : public testing::WithParamInterface<EltwiseLayerCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(testing::TestParamInfo<EltwiseLayerCPUTestParamsSet> obj) {
        subgraph::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = obj.param;

        std::ostringstream result;
        result << subgraph::EltwiseLayerTest::getTestCaseName(testing::TestParamInfo<subgraph::EltwiseTestParams>(
                basicParamsSet, 0));
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);

        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            bool isReal = funcInput.get_element_type().is_real();
            switch (eltwiseType) {
                case ngraph::helpers::EltwiseTypes::POWER:
                case ngraph::helpers::EltwiseTypes::MOD:
                case ngraph::helpers::EltwiseTypes::FLOOR_MOD:
                    tensor = isReal ?
                             ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2, 2, 8) :
                             ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 4, 2);
                    break;
                case ngraph::helpers::EltwiseTypes::DIVIDE:
                    tensor = isReal ?
                             ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2, 2, 8) :
                             ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 100, 101);
                    break;
                case ngraph::helpers::EltwiseTypes::ERF:
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 6, -3);
                    break;
                default:
                    if (funcInput.get_element_type().is_real()) {
                        tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 80, 0, 8);
                    } else {
                        tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                    }
                    break;
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }


    void SetUp() override {
        subgraph::EltwiseTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        fusingSpecificParams fusingParams;
        std::tie(basicParamsSet, cpuParams, fusingParams) = this->GetParam();

        std::vector<InputShape> shapes;
        ElementType netType;
        ngraph::helpers::InputLayerType secondaryInputType;
        CommonTestUtils::OpType opType;
        Config additional_config;
        std::tie(shapes, eltwiseType, secondaryInputType, opType, netType, inType, outType, targetDevice, configuration) = basicParamsSet;

        if (ElementType::bf16 == netType) {
            rel_threshold = 2e-2f;
        }

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        selectedType = makeSelectedTypeStr(getPrimitiveType(), netType);

        shapes.resize(2);
        switch (opType) {
            case CommonTestUtils::OpType::SCALAR: {
                std::vector<ngraph::Shape> identityShapes(shapes[0].second.size(), {1});
                shapes[1] = {{}, identityShapes};
                break;
            }
            case CommonTestUtils::OpType::VECTOR:
                if (shapes[1].second.empty()) {
                    shapes[1] = shapes[0];
                }
                break;
            default:
                FAIL() << "Unsupported Secondary operation type";
        }

        init_input_shapes(shapes);

        configuration.insert(additional_config.begin(), additional_config.end());
        auto parameters = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.front()});

        std::shared_ptr<ngraph::Node> secondaryInput;
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            secondaryInput = ngraph::builder::makeDynamicParams(netType, {inputDynamicShapes.back()}).front();
            parameters.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(secondaryInput));
        } else {
            auto pShape = inputDynamicShapes.back();
            ngraph::Shape shape;
            if (pShape.is_static()) {
                shape = pShape.get_shape();
            } else {
                ASSERT_TRUE(pShape.rank().is_static());
                shape = std::vector<size_t>(pShape.rank().get_length(), 1);
                for (size_t i = 0; i < pShape.size(); ++i) {
                    if (pShape[i].is_static()) {
                        shape[i] = pShape[i].get_length();
                    }
                }
            }
            if (eltwiseType == ngraph::helpers::EltwiseTypes::DIVIDE ||
                eltwiseType == ngraph::helpers::EltwiseTypes::MOD) {
                std::vector<float> data(ngraph::shape_size(shape));
                data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(ngraph::shape_size(shape), 10, 2);
                secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
            } else if (eltwiseType == ngraph::helpers::EltwiseTypes::FLOOR_MOD) {
                auto negative_data_size = ngraph::shape_size(shape) / 2;
                auto positive_data_size = ngraph::shape_size(shape) - negative_data_size;
                std::vector<float> negative_data(negative_data_size);
                std::vector<float> data(positive_data_size);
                negative_data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(negative_data_size, -10, -2);
                data = NGraphFunctions::Utils::generateVector<ngraph::element::Type_t::f32>(positive_data_size, 10, 2);
                data.insert(data.end(), negative_data.begin(), negative_data.end());
                secondaryInput = ngraph::builder::makeConstant(netType, shape, data);
            } else if (eltwiseType == ngraph::helpers::EltwiseTypes::POWER) {
                secondaryInput = ngraph::builder::makeConstant<float>(netType, shape, {}, true, 3);
            } else {
                secondaryInput = ngraph::builder::makeConstant<float>(netType, shape, {}, true);
            }
        }

        auto eltwise = ngraph::builder::makeEltwise(parameters[0], secondaryInput, eltwiseType);

        function = makeNgraphFunction(netType, parameters, eltwise, "Eltwise");
    }

private:
    ngraph::helpers::EltwiseTypes eltwiseType;
};

TEST_P(EltwiseLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckPluginRelatedResults(compiledModel, "Eltwise");
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

ov::AnyMap additional_config;

std::vector<ElementType> netType = {ElementType::bf16, ElementType::f32};

std::vector<CPUSpecificParams> cpuParams_4D = {
        CPUSpecificParams({nChw16c, nChw16c}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
};

std::vector<CPUSpecificParams> cpuParams_5D = {
        CPUSpecificParams({nCdhw16c, nCdhw16c}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
};

const std::vector<fusingSpecificParams> fusingParamsSet{
    emptyFusingSpec,
    // eltwise
    fusingSigmoid,
    fusingPRelu1D,
    // depthwise
    fusingReluScaleShift,
    // fake quantize
    fusingFakeQuantizePerTensorRelu,
    fusingFakeQuantizePerChannelRelu,
    fusingFQPerChannelSigmoidFQPerChannel
};

std::vector<std::vector<ov::Shape>> inShapes_4D = {
        {{2, 4, 4, 1}},
        {{2, 17, 5, 4}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
};

const auto params_4D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder, EltwiseLayerCPUTest, params_4D, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inShapes_4D_fusing = {
        {{2, 4, 4, 1}},
        {{2, 17, 5, 4}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
};

const auto params_4D_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_fusing)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(cpuParams_4D),
        ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Fusing, EltwiseLayerCPUTest, params_4D_fusing, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D)),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_emptyCPUSpec, EltwiseLayerCPUTest, params_4D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inShapes_5D = {
        {{2, 4, 3, 4, 1}},
        {{2, 17, 7, 5, 4}},
        {{2, 17, 6, 5, 4}, {1, 17, 6, 1, 1}},
        {{2, 17, 6, 5, 1}, {1, 17, 1, 1, 4}},
};

const auto params_5D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder, EltwiseLayerCPUTest, params_5D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D)),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D, EltwiseLayerCPUTest, params_5D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inShapes_4D_Blocked_Planar = {
        {{2, 17, 31, 3}, {2, 1, 31, 3}},
        {{2, 17, 5, 1}, {2, 1, 1, 4}},
};

std::vector<CPUSpecificParams> cpuParams_4D_Blocked_Planar = {
        CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {}),
};

const auto params_4D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_Blocked_Planar)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Planar)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Planar, EltwiseLayerCPUTest, params_4D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> inShapes_4D_Planar_Blocked = {
        {{2, 1, 31, 3}, {2, 17, 31, 3}},
        {{2, 1, 1, 4}, {2, 17, 5, 1}},
};

std::vector<CPUSpecificParams> cpuParams_4D_Planar_Blocked = {
        CPUSpecificParams({nchw, nChw16c}, {nChw16c}, {}, {}),
};

const auto params_4D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_Planar_Blocked)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar_Blocked)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Planar_Blocked, EltwiseLayerCPUTest, params_4D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> inShapes_5D_Blocked_Planar = {
        {{2, 17, 31, 4, 3}, {2, 1, 31, 1, 3}},
        {{2, 17, 5, 3, 1}, {2, 1, 1, 3, 4}},
};

std::vector<CPUSpecificParams> cpuParams_5D_Blocked_Planar = {
        CPUSpecificParams({nCdhw16c, ncdhw}, {nCdhw16c}, {}, {}),
};

const auto params_5D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_Blocked_Planar)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Planar)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Blocked_Planar, EltwiseLayerCPUTest, params_5D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<ngraph::Shape>> inShapes_5D_Planar_Blocked = {
        {{2, 1, 31, 1, 3}, {2, 17, 31, 4, 3}},
        {{2, 1, 1, 3, 4}, {2, 17, 5, 3, 1}},
};

std::vector<CPUSpecificParams> cpuParams_5D_Planar_Blocked = {
        CPUSpecificParams({ncdhw, nCdhw16c}, {nCdhw16c}, {}, {}),
};

const auto params_5D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_Planar_Blocked)),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Planar_Blocked)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Planar_Blocked, EltwiseLayerCPUTest, params_5D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);


std::vector<std::vector<ngraph::Shape>> inShapes_4D_1D = {
        {{2, 17, 5, 4}, {4}},
        {{1, 3, 3, 3}, {3}},
};

std::vector<CPUSpecificParams> cpuParams_4D_1D_Constant_mode = {
        CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, nchw}, {nchw}, {}, {})
};

const auto params_4D_1D_constant_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D)),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Constant_mode)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Constant, EltwiseLayerCPUTest, params_4D_1D_constant_mode, EltwiseLayerCPUTest::getTestCaseName);

std::vector<CPUSpecificParams> cpuParams_4D_1D_Parameter_mode = {
        CPUSpecificParams({nChw16c, x}, {nChw16c}, {}, {}),
        CPUSpecificParams({nhwc, x}, {nhwc}, {}, {}),
        CPUSpecificParams({nchw, x}, {nchw}, {}, {})
};

const auto params_4D_1D_parameter_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D)),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Parameter_mode)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Parameter, EltwiseLayerCPUTest, params_4D_1D_parameter_mode, EltwiseLayerCPUTest::getTestCaseName);

std::vector<std::vector<ngraph::Shape>> inShapes_5D_1D = {
        {{2, 17, 5, 4, 10}, {10}},
        {{1, 3, 3, 3, 3}, {3}},
};

std::vector<CPUSpecificParams> cpuParams_5D_1D_constant = {
        CPUSpecificParams({nCdhw16c, ncdhw}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc, ndhwc}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, ncdhw}, {ncdhw}, {}, {})
};

const auto params_5D_1D_constant = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D)),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_constant)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Constant, EltwiseLayerCPUTest, params_5D_1D_constant, EltwiseLayerCPUTest::getTestCaseName);

std::vector<CPUSpecificParams> cpuParams_5D_1D_parameter = {
        CPUSpecificParams({nCdhw16c, x}, {nCdhw16c}, {}, {}),
        CPUSpecificParams({ndhwc, x}, {ndhwc}, {}, {}),
        CPUSpecificParams({ncdhw, x}, {ncdhw}, {}, {})
};

const auto params_5D_1D_parameter = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D)),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_parameter)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Parameter, EltwiseLayerCPUTest, params_5D_1D_parameter, EltwiseLayerCPUTest::getTestCaseName);


std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesBinDyn = {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
};

//// ============================================ 4D ============================================
std::vector<std::vector<InputShape>> inShapes_4D_dyn_const = {
    {
        {
            // dynamic
            {3, 2, -1, -1},
            // target
            {
                {3, 2, 1, 1},
                {3, 2, 5, 1},
                {3, 2, 1, 6},
                {3, 2, 4, 11},
            }
        }
    },
    {
        {
           // dynamic
           {{1, 10}, 2, 5, 6},
           // target
           {
               {3, 2, 5, 6},
               {1, 2, 5, 6},
               {2, 2, 5, 6},
           }
        }
    },
};

const auto params_4D_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_dyn_const),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_const, EltwiseLayerCPUTest, params_4D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_dyn_param = {
    {
        // dynamic
        {-1, {2, 15}, -1, -1},
        // target
        {
            {3, 2, 1, 1},
            {1, 7, 5, 1},
            {3, 3, 4, 11},
        }
    },
    {
        // dynamic
        {-1, {2, 25}, -1, -1},
        // target
        {
            {1, 2, 5, 1},
            {3, 7, 1, 10},
            {3, 3, 4, 11}
        }
    }
};

const auto params_4D_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_param, EltwiseLayerCPUTest, params_4D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_dyn_param_fusing = {
    {
        // dynamic
        {-1, 7, -1, -1},
        // target
        {
            {3, 7, 1, 1},
            {1, 7, 5, 1},
            {3, 7, 1, 1},
            {3, 7, 4, 11},
        }
    },
    {
        // dynamic
        {-1, 7, -1, -1},
        // target
        {
            {1, 7, 5, 1},
            {3, 7, 1, 10},
            {1, 7, 5, 1},
            {3, 7, 4, 11}
        }
    }
};

const auto params_4D_dyn_param_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param_fusing),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(cpuParams_4D),
        ::testing::ValuesIn(fusingParamsSet));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_dyn_param_fusing, EltwiseLayerCPUTest, params_4D_dyn_param_fusing, EltwiseLayerCPUTest::getTestCaseName);

//// ============================================ 5D ============================================
std::vector<InputShape> inShapes_5D_dyn_const = {
    {
        // dynamic
        {3, 2, -1, -1, -1},
        // target
        {
            {3, 2, 1, 1, 1},
            {3, 2, 5, 1, 7},
            {3, 2, 1, 6, 1},
            {3, 2, 4, 11, 2},
        }
    },
};

const auto params_5D_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_const),
                ::testing::ValuesIn(eltwiseOpTypesBinInp),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_const, EltwiseLayerCPUTest, params_5D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

std::vector<InputShape> inShapes_5D_dyn_param = {
    {
        // dynamic
        {-1, {2, 15}, -1, -1, -1},
        // target
        {
            {3, 2, 1, 1, 1},
            {1, 7, 5, 1, 12},
            {3, 3, 4, 11, 6},
        }
    },
    {
        // dynamic
        {-1, {2, 25}, -1, -1, -1},
        // target
        {
            {1, 2, 5, 1, 5},
            {3, 7, 1, 10, 1},
            {3, 3, 4, 11, 6}
        }
    }
};

const auto params_5D_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_param),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes),
                ::testing::ValuesIn(netType),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D)),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_param, EltwiseLayerCPUTest, params_5D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions