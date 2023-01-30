// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/pad.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov;
using namespace test;

namespace CPULayerTestsDefinitions {

using PadLayerCPUTestParamSet = std::tuple<
        InputShape,                                     // Input shape
        ngraph::helpers::InputLayerType,                // Secondary input types
        ElementType,                                    // Input element type
        std::vector<int64_t>,                           // padsBegin
        std::vector<int64_t>,                           // padsEnd
        float,                                          // argPadValue
        ngraph::helpers::PadMode,                       // padMode
        CPUSpecificParams
>;

class PadLayerCPUTest : public testing::WithParamInterface<PadLayerCPUTestParamSet>,
                        virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PadLayerCPUTestParamSet> obj) {
        InputShape shapes;
        ngraph::helpers::InputLayerType secondaryInputType;
        ElementType elementType;
        std::vector<int64_t> padsBegin, padsEnd;
        ngraph::helpers::PadMode padMode;
        float argPadValue;
        CPUSpecificParams cpuParams;
        std::tie(shapes, secondaryInputType, elementType, padsBegin, padsEnd, argPadValue, padMode, cpuParams) = obj.param;

        std::ostringstream results;
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
        results << "secondaryInputType=" << secondaryInputType << "_";
        results << "Prc=" << elementType << "_";
        results << "padsBegin=" << CommonTestUtils::vec2str(padsBegin) << "_";
        results << "padsEnd=" << CommonTestUtils::vec2str(padsEnd) << "_";
        if (padMode == ngraph::helpers::PadMode::CONSTANT) {
            results << "Value=" << argPadValue << "_";
        }
        results << "PadMode=" << padMode << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

protected:
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        std::vector<void*> inputValues = {padsBegin.data(), padsEnd.data(), &padValue};

        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 10, 1, 1);
            } else {
                if (funcInput.get_node()->get_friendly_name() == "pad_value")
                    tensor = ov::Tensor{funcInput.get_element_type(), ov::Shape{}, &padValue};
                else
                    tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i], inputValues[i-1]};
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
    void SetUp() override {
        InputShape shapes;
        ngraph::helpers::InputLayerType secondaryInputType;
        ngraph::helpers::PadMode padMode;
        ov::element::Type dataType;
        CPUSpecificParams cpuParams;
        std::tie(shapes, secondaryInputType, dataType, padsBegin, padsEnd, padValue, padMode, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = makeSelectedTypeStr("ref", dataType);
        targetDevice = CommonTestUtils::DEVICE_CPU;
        init_input_shapes({shapes});
        for (auto& targetShapes : targetStaticShapes) {
            targetShapes.push_back({padsBegin.size()});
            targetShapes.push_back({padsEnd.size()});
            targetShapes.push_back({});
        }
        auto params = ngraph::builder::makeDynamicParams(dataType, inputDynamicShapes);
        std::shared_ptr<ngraph::Node> pad;
        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            ov::Shape inShape = {padsBegin.size()};

            auto beginNode = std::make_shared<ngraph::opset1::Parameter>(ov::element::i64, inShape);
            auto endNode = std::make_shared<ngraph::opset1::Parameter>(ov::element::i64, inShape);
            std::shared_ptr<ngraph::opset1::Parameter> valueNode = nullptr;
            params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(beginNode));
            params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(endNode));
            if (padMode == ngraph::helpers::PadMode::CONSTANT) {
                valueNode = std::make_shared<ngraph::opset1::Parameter>(dataType, ov::Shape{});
                params.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(valueNode));
                params.back()->set_friendly_name("pad_value");
            }
            pad = ngraph::builder::makePad(params[0], beginNode, endNode, valueNode, padMode);
        } else {
            pad = ngraph::builder::makePad(params[0], padsBegin, padsEnd, padValue, padMode);
        }
        function = makeNgraphFunction(inType, params, pad, "Pad");
    }
    std::vector<int64_t> padsBegin;  // padsBegin
    std::vector<int64_t> padsEnd;    // padsEnd
    float padValue;                  // argPadValue
};

TEST_P(PadLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Pad");
}

namespace {


const auto cpuParams_nChw16c = CPUSpecificParams {{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams {{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams {{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams {{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams {{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams {{ndhwc}, {ndhwc}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams {{nchw}, {nchw}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams {{ncdhw}, {ncdhw}, {}, {}};

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i8
};

const std::vector<ngraph::helpers::InputLayerType> inputLayerTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
};

const std::vector<ngraph::helpers::InputLayerType> inputLayerTypesBlocked = {
    ngraph::helpers::InputLayerType::CONSTANT,
};

const std::vector<float> argPadValue = {0.f, 2.5f, -1.f};

const std::vector<ngraph::helpers::PadMode> padMode = {
        ngraph::helpers::PadMode::EDGE,
        ngraph::helpers::PadMode::REFLECT,
        ngraph::helpers::PadMode::SYMMETRIC
};

/* *======================* Static Shapes Tests 4D *======================* */

const std::vector<std::vector<int64_t>> padsBegin4DConstBlocked_Smoke = {{0, 0, 1, 3}, {2, 16, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DConstBlocked_Smoke   = {{0, 0, 2, 1}, {2, 0, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin4DBlocked_Smoke = {{0, 0, 1, 3}, {2, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DBlocked_Smoke   = {{0, 0, 2, 1}, {2, 0, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin4D_Smoke = {{0, 1, 1, 1}, {0, 2, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4D_Smoke   = {{0, 2, 1, 1}, {0, 0, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4DConstBlocked_Full = {{0, 0, 0, 0}, {0, 0, 1, 3}, {2, 16, 1, 0}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DConstBlocked_Full   = {{0, 0, 0, 0}, {0, 0, 2, 1}, {2, 0, 0, 1}, {1, 32, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4DBlocked_Full = {{0, 0, 0, 0}, {0, 0, 1, 3}, {2, 0, 1, 0}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DBlocked_Full   = {{0, 0, 0, 0}, {0, 0, 2, 1}, {2, 0, 0, 1}, {1, 0, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4D_Full = {{0, 0, 0, 0}, {0, 1, 1, 1}, {0, 2, 1, 0}, {0, 0, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D_Full   = {{0, 0, 0, 0}, {0, 2, 1, 1}, {0, 0, 2, 0}, {1, 1, 0, 0}};

const std::vector<CPUSpecificParams> CPUParams4DBlocked = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
};

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DConstBlocked_Smoke),
                ::testing::ValuesIn(padsEnd4DConstBlocked_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 10, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DBlocked_Smoke),
                ::testing::ValuesIn(padsEnd4DBlocked_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 10, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DBlocked_Smoke),
                ::testing::ValuesIn(padsEnd4DBlocked_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad4DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DConstBlocked_Full),
                ::testing::ValuesIn(padsEnd4DConstBlocked_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad4DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Full),
                ::testing::ValuesIn(padsEnd4D_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 10, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DBlocked_Full),
                ::testing::ValuesIn(padsEnd4DBlocked_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad4D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 10, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DBlocked_Full),
                ::testing::ValuesIn(padsEnd4DBlocked_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

/* *======================* *=====================* *======================* */

/* *======================* Dynamic Shapes Tests 4D *======================* */

const std::vector<InputShape> inputShapesDynamic4D = {
        {{-1, -1, -1, -1},                                    // dynamic
         {{5, 36, 5, 5}, {3, 16, 10, 5}, {3, 24, 10, 10}}},   // target

        {{-1, 32, -1, -1},                                    // dynamic
         {{5, 32, 5, 5}, {5, 32, 5, 8}, {3, 32, 8, 8}}},      // target

        {{{1, 5}, {16, 32}, {1, 16}, {1, 16}},                // dynamic
         {{3, 16, 5, 5}, {5, 24, 5, 8}, {3, 32, 8, 8}}},      // target
};

const std::vector<CPUSpecificParams> CPUParams4DDynamic = {
        cpuParams_nhwc,
        cpuParams_nchw
};

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic4DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams4DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic4DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic4D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DConstBlocked_Smoke),
                ::testing::ValuesIn(padsEnd4DConstBlocked_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic4D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams4DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic4D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DBlocked_Smoke),
                ::testing::ValuesIn(padsEnd4DBlocked_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic4DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Full),
                ::testing::ValuesIn(padsEnd4D_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams4DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic4DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic4D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DConstBlocked_Full),
                ::testing::ValuesIn(padsEnd4DConstBlocked_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic4D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Full),
                ::testing::ValuesIn(padsEnd4D_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams4DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic4D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DBlocked_Full),
                ::testing::ValuesIn(padsEnd4DBlocked_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams4DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

/* *======================* *=====================* *======================* */

/* *======================* Static Shapes Tests 5D *======================* */

const std::vector<std::vector<int64_t>> padsBegin5DConstBlocked_Smoke = {{0, 0, 1, 1, 0}, {2, 32, 1, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DConstBlocked_Smoke   = {{1, 16, 1, 1, 0}, {0, 0, 0, 1, 0}};

const std::vector<std::vector<int64_t>> padsBegin5DBlocked_Smoke = {{0, 0, 1, 1, 0}, {2, 0, 1, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DBlocked_Smoke   = {{1, 0, 1, 1, 0}, {0, 0, 0, 1, 0}};

const std::vector<std::vector<int64_t>> padsBegin5D_Smoke = {{0, 0, 2, 0, 0}, {1, 1, 1, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5D_Smoke   = {{0, 0, 1, 0, 0}, {1, 0, 1, 1, 2}};

const std::vector<std::vector<int64_t>> padsBegin5DConstBlocked_Full = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 0}, {2, 32, 1, 1, 0}, {0, 0, 1, 3, 1}, {0, 0, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DConstBlocked_Full   = {{0, 0, 0, 0, 0}, {1, 16, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}, {0, 0, 1, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin5DBlocked_Full = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 0}, {2, 0, 1, 1, 0}, {0, 0, 1, 3, 1}, {0, 0, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DBlocked_Full   = {{0, 0, 0, 0, 0}, {1, 0, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 1}, {0, 0, 1, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin5D_Full = {{0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {1, 1, 1, 1, 0}, {2, 0, 1, 0, 1}, {0, 2, 1, 3, 1}};
const std::vector<std::vector<int64_t>> padsEnd5D_Full   = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {1, 0, 1, 1, 2}, {2, 2, 0, 1, 0}, {1, 1, 2, 0, 1}};

const std::vector<CPUSpecificParams> CPUParams5DBlocked = {
        cpuParams_nCdhw16c,
        cpuParams_nCdhw8c,
};

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DConstBlocked_Smoke),
                ::testing::ValuesIn(padsEnd5DConstBlocked_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::Values(cpuParams_ndhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DBlocked_Smoke),
                ::testing::ValuesIn(padsEnd5DBlocked_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad5D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::Values(cpuParams_ndhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad5DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DConstBlocked_Full),
                ::testing::ValuesIn(padsEnd5DConstBlocked_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad5DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Full),
                ::testing::ValuesIn(padsEnd5D_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::Values(cpuParams_ndhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad5DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DBlocked_Full),
                ::testing::ValuesIn(padsEnd5DBlocked_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad5D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 16, 5, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Full),
                ::testing::ValuesIn(padsEnd5D_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::Values(cpuParams_ndhwc)),
        PadLayerCPUTest::getTestCaseName
);

/* *======================* *=====================* *======================* */

/* *======================* Dynamic Shapes Tests 5D *======================* */

const std::vector<InputShape> inputShapesDynamic5D = {
        {{-1, -1, -1, -1, -1},                                            // dynamic
         {{5, 36, 5, 5, 5}, {3, 16, 8, 5, 7}, {3, 24, 10, 10, 10}}},      // target

        {{-1, 32, -1, -1, -1},                                            // dynamic
         {{5, 32, 5, 5, 5}, {3, 32, 8, 5, 7}, {3, 32, 10, 10, 10}}},      // target

        {{{1, 5}, {16, 32}, {1, 16}, {1, 16}, {1, 16}},                   // dynamic
         {{3, 16, 5, 5, 5}, {3, 24, 8, 5, 7}, {4, 32, 10, 10, 10}}},      // target
};

const std::vector<CPUSpecificParams> CPUParams5DDynamic = {
        cpuParams_ndhwc,
        cpuParams_ncdhw
};

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic5DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams5DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic5DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic5D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DConstBlocked_Smoke),
                ::testing::ValuesIn(padsEnd5DConstBlocked_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic5D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams5DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPadDynamic5DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic5D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DBlocked_Smoke),
                ::testing::ValuesIn(padsEnd5DBlocked_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic5DConst,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Full),
                ::testing::ValuesIn(padsEnd5D_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams5DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic5DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic5D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DConstBlocked_Full),
                ::testing::ValuesIn(padsEnd5DConstBlocked_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic5D,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputLayerTypes),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Full),
                ::testing::ValuesIn(padsEnd5D_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams5DDynamic)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPadDynamic5DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::Values(inputShapesDynamic5D[1]),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5DBlocked_Full),
                ::testing::ValuesIn(padsEnd5DBlocked_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode),
                ::testing::ValuesIn(CPUParams5DBlocked)),
        PadLayerCPUTest::getTestCaseName
);

/* *======================* *=====================* *======================* */

} // namespace
} // namespace CPULayerTestsDefinitions

