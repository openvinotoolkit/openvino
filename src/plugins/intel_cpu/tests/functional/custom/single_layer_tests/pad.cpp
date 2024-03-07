// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/pad.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/test_enums.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using PadLayerCPUTestParamSet = std::tuple<
        InputShape,                                     // Input shape
        ov::test::utils::InputLayerType,                // Secondary input types
        ElementType,                                    // Input element type
        std::vector<int64_t>,                           // padsBegin
        std::vector<int64_t>,                           // padsEnd
        float,                                          // argPadValue
        ov::op::PadMode,                                // padMode
        CPUSpecificParams
>;

class PadLayerCPUTest : public testing::WithParamInterface<PadLayerCPUTestParamSet>,
                        virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PadLayerCPUTestParamSet> obj) {
        InputShape shapes;
        ov::test::utils::InputLayerType secondaryInputType;
        ElementType elementType;
        std::vector<int64_t> padsBegin, padsEnd;
        ov::op::PadMode padMode;
        float argPadValue;
        CPUSpecificParams cpuParams;
        std::tie(shapes, secondaryInputType, elementType, padsBegin, padsEnd, argPadValue, padMode, cpuParams) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "secondaryInputType=" << secondaryInputType << "_";
        results << "Prc=" << elementType << "_";
        results << "padsBegin=" << ov::test::utils::vec2str(padsBegin) << "_";
        results << "padsEnd=" << ov::test::utils::vec2str(padsEnd) << "_";
        if (padMode == ov::op::PadMode::CONSTANT) {
            results << "Value=" << argPadValue << "_";
        }
        results << "PadMode=" << padMode << "_";
        results << CPUTestsBase::getTestCaseName(cpuParams);

        return results.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<void*> inputValues = {padsBegin.data(), padsEnd.data(), &padValue};

        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 1;
                in_data.range = 10;
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
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
        ov::test::utils::InputLayerType secondaryInputType;
        ov::op::PadMode padMode;
        ElementType dataType;
        CPUSpecificParams cpuParams;
        std::tie(shapes, secondaryInputType, dataType, padsBegin, padsEnd, padValue, padMode, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType = makeSelectedTypeStr("ref", dataType);
        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});
        for (auto& targetShapes : targetStaticShapes) {
            targetShapes.push_back({padsBegin.size()});
            targetShapes.push_back({padsEnd.size()});
            targetShapes.push_back({});
        }
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(dataType, shape));
        }
        std::shared_ptr<ov::Node> pad;
        if (secondaryInputType == ov::test::utils::InputLayerType::PARAMETER) {
            ov::Shape inShape = {padsBegin.size()};

            auto beginNode = std::make_shared<ov::op::v0::Parameter>(ElementType::i64, inShape);
            auto endNode = std::make_shared<ov::op::v0::Parameter>(ElementType::i64, inShape);
            std::shared_ptr<ov::op::v0::Parameter> valueNode = nullptr;
            params.push_back(beginNode);
            params.push_back(endNode);
            if (padMode == ov::op::PadMode::CONSTANT) {
                valueNode = std::make_shared<ov::op::v0::Parameter>(dataType, ov::Shape{});
                params.push_back(valueNode);
                params.back()->set_friendly_name("pad_value");
                pad = std::make_shared<ov::op::v12::Pad>(params[0], beginNode, endNode, valueNode, padMode);
            } else {
                pad = std::make_shared<ov::op::v12::Pad>(params[0], beginNode, endNode, padMode);
            }
        } else {
            auto padsBeginNode = std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape{padsBegin.size()}, padsBegin.data());
            auto padsEndNode = std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape{padsEnd.size()}, padsEnd.data());
            auto argPadValueNode = std::make_shared<ov::op::v0::Constant>(params[0]->get_element_type(), ov::Shape{}, &padValue);

            pad = std::make_shared<ov::op::v12::Pad>(params[0], padsBeginNode, padsEndNode, argPadValueNode, padMode);
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

const std::vector<ov::test::utils::InputLayerType> inputLayerTypes = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER
};

const std::vector<ov::test::utils::InputLayerType> inputLayerTypesBlocked = {
    ov::test::utils::InputLayerType::CONSTANT,
};

const std::vector<float> argPadValue = {0.f, 2.5f};

const std::vector<ov::op::PadMode> padMode = {
        ov::op::PadMode::EDGE,
        ov::op::PadMode::REFLECT,
        ov::op::PadMode::SYMMETRIC
};

/* *======================* Static Shapes Tests 4D *======================* */

const std::vector<std::vector<int64_t>> padsBegin4DConstBlocked_Smoke = {{0, 0, 1, 3}, {2, 16, 1, 0}, {2, -16, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DConstBlocked_Smoke   = {{0, 0, 2, -1}, {2, 0, 0, 1}, {2, 16, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin4DBlocked_Smoke = {{0, 0, -1, 3}, {2, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DBlocked_Smoke   = {{0, 0, 2, 1}, {2, 0, 0, -1}};

const std::vector<std::vector<int64_t>> padsBegin4D_Smoke = {{0, 1, 1, 1}, {0, 2, 1, 0}, {0, 0, -2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4D_Smoke   = {{0, 2, 1, 1}, {0, 0, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4DConstBlocked_Full = {{0, 0, 0, 0}, {0, 0, 1, -3}, {2, 16, 1, 0}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DConstBlocked_Full   = {{0, 0, 0, 0}, {0, 0, 2, 1}, {2, 0, 0, 1}, {1, -16, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4DBlocked_Full = {{0, 0, 0, 0}, {0, 0, 1, 3}, {2, 0, 1, 0}, {0, 0, -2, 0}};
const std::vector<std::vector<int64_t>> padsEnd4DBlocked_Full   = {{0, 0, 0, 0}, {0, 0, -2, -1}, {2, 0, 0, 1}, {1, 0, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4D_Full = {{0, 0, -1, 0}, {0, 0, 1, 0}, {0, 2, 0, 0}, {0, -2, 0, 0}, {0, 0, 0, 2}, {0, 0, 0, -2}};
const std::vector<std::vector<int64_t>> padsEnd4D_Full   = {{0, 0, -2, 0}, {0, 0, 2, 0}, {0, 1, 0, 0}, {0, -2, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, -2}};

const std::vector<CPUSpecificParams> CPUParams4DBlocked = {
        cpuParams_nChw16c,
        cpuParams_nChw8c,
};

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DConstBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 32, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DConstBlocked_Smoke),
                ::testing::ValuesIn(padsEnd4DConstBlocked_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_CPUPad4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 32, 10, 5}})),
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
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 32, 5, 5}})),
                ::testing::ValuesIn(inputLayerTypesBlocked),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4DConstBlocked_Full),
                ::testing::ValuesIn(padsEnd4DConstBlocked_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
                ::testing::Values(cpuParams_nhwc)),
        PadLayerCPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        CPUPad4DBlocked,
        PadLayerCPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation({{3, 32, 10, 5}})),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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

const std::vector<std::vector<int64_t>> padsBegin5DConstBlocked_Smoke = {{0, 0, 1, 1, 0}, {2, 32, 1, -1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DConstBlocked_Smoke   = {{1, 16, -1, 1, 0}, {0, 0, 0, 1, 0}};

const std::vector<std::vector<int64_t>> padsBegin5DBlocked_Smoke = {{0, 0, -1, 1, 0}, {2, 0, 1, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DBlocked_Smoke   = {{1, 0, 1, 1, 0}, {0, 0, 0, -1, 0}};

const std::vector<std::vector<int64_t>> padsBegin5D_Smoke = {{0, 0, -2, 0, 0}, {1, 1, 1, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5D_Smoke   = {{0, 0, 1, 0, 0}, {1, 0, 1, 1, -2}};

const std::vector<std::vector<int64_t>> padsBegin5DConstBlocked_Full = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 0}, {2, 32, 1, 1, 0}, {0, 0, 1, 3, 1}, {0, 0, 0, -1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DConstBlocked_Full   = {{0, 0, 0, 0, 0}, {1, 16, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, -1, 1}, {0, 0, 1, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin5DBlocked_Full = {{0, 0, 0, 0, 0}, {0, 0, 1, 1, 0}, {2, 0, 1, -1, 0}, {0, 0, 1, 3, 1}, {0, 0, 0, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5DBlocked_Full   = {{0, 0, 0, 0, 0}, {1, 0, 1, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, -1, -1}, {0, 0, 1, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin5D_Full = {{0, 0, 0, 0, 0}, {0, 0, -2, 0, 0}, {1, 1, 1, 1, 0}, {2, 0, 1, 0, -1}, {0, 2, 1, 3, 1}};
const std::vector<std::vector<int64_t>> padsEnd5D_Full   = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {1, 0, 1, 1, 2}, {2, 2, 0, 1, 0}, {1, 1, 2, 0, -1}};

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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
                ::testing::Values(ov::op::PadMode::CONSTANT),
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
}  // namespace test
}  // namespace ov

