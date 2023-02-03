// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/pad.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace InferenceEngine;
using namespace ov;
using namespace test;

namespace GPULayerTestsDefinitions {

using PadLayerGPUTestParamSet = std::tuple<
        InputShape,                                     // Input shape
        ElementType,                                    // Input element type
        std::vector<int64_t>,                           // padsBegin
        std::vector<int64_t>,                           // padsEnd
        float,                                          // argPadValue
        ngraph::helpers::PadMode                        // padMode
>;

class PadLayerGPUTest : public testing::WithParamInterface<PadLayerGPUTestParamSet>,
                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<PadLayerGPUTestParamSet> obj) {
        InputShape shapes;
        ElementType elementType;
        std::vector<int64_t> padsBegin, padsEnd;
        ngraph::helpers::PadMode padMode;
        float argPadValue;
        std::tie(shapes, elementType, padsBegin, padsEnd, argPadValue, padMode) = obj.param;

        std::ostringstream results;
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << CommonTestUtils::vec2str(item) << "_";
        }
        results << "Prc=" << elementType << "_";
        results << "padsBegin=" << CommonTestUtils::vec2str(padsBegin) << "_";
        results << "padsEnd=" << CommonTestUtils::vec2str(padsEnd) << "_";
        if (padMode == ngraph::helpers::PadMode::CONSTANT) {
            results << "Value=" << argPadValue << "_";
        }
        results << "PadMode=" << padMode << "_";

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        std::vector<int64_t> padsBegin, padsEnd;
        ngraph::helpers::PadMode padMode;
        float argPadValue;
        std::tie(shapes, inType, padsBegin, padsEnd, argPadValue, padMode) = this->GetParam();

        targetDevice = CommonTestUtils::DEVICE_GPU;
        init_input_shapes({shapes});

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto pad = ngraph::builder::makePad(params[0], padsBegin, padsEnd, argPadValue, padMode);

        ngraph::ResultVector results;
        for (int i = 0; i < pad->get_output_size(); ++i) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(pad->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, params, "Pad");
    }
};

TEST_P(PadLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {

const std::vector<ElementType> inputPrecisions = {
        ElementType::f32,
        ElementType::i8
};

const std::vector<float> argPadValue = {0.f, 2.5f, -1.f};

const std::vector<ngraph::helpers::PadMode> padMode = {
        ngraph::helpers::PadMode::EDGE,
        ngraph::helpers::PadMode::REFLECT,
        ngraph::helpers::PadMode::SYMMETRIC
};

/* *======================* Dynamic Shapes Tests 2D *======================* */

const std::vector<InputShape> inputShapesDynamic2D = {
        {{-1, -1},              // dynamic
         {{5, 36}, {3, 16}}},   // target

        {{-1, 32},              // dynamic
         {{5, 32}}},            // target

        {{{1, 5}, {16, 32}},    // dynamic
         {{3, 16}, {5, 24}}},   // target
};

const std::vector<std::vector<int64_t>> padsBegin2D_Smoke = {{0, 1}, {0, 2}};
const std::vector<std::vector<int64_t>> padsEnd2D_Smoke   = {{0, 2}, {0, 0}};

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic2DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic2D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin2D_Smoke),
                ::testing::ValuesIn(padsEnd2D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic2D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic2D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin2D_Smoke),
                ::testing::ValuesIn(padsEnd2D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
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

const std::vector<std::vector<int64_t>> padsBegin4D_Smoke = {{0, 1, 1, 1}, {0, 2, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd4D_Smoke   = {{0, 2, 1, 1}, {0, 0, 2, 0}};

const std::vector<std::vector<int64_t>> padsBegin4D_Full = {{0, 0, 0, 0}, {0, 1, 1, 1}, {0, 2, 1, 0}, {0, 0, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D_Full   = {{0, 0, 0, 0}, {0, 2, 1, 1}, {0, 0, 2, 0}, {1, 1, 0, 0}};

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic4DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic4D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Smoke),
                ::testing::ValuesIn(padsEnd4D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        GPUPadDynamic4DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Full),
                ::testing::ValuesIn(padsEnd4D_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        GPUPadDynamic4D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic4D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin4D_Full),
                ::testing::ValuesIn(padsEnd4D_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
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

const std::vector<std::vector<int64_t>> padsBegin5D_Smoke = {{0, 0, 2, 0, 0}, {1, 1, 1, 1, 0}};
const std::vector<std::vector<int64_t>> padsEnd5D_Smoke   = {{0, 0, 1, 0, 0}, {1, 0, 1, 1, 2}};

const std::vector<std::vector<int64_t>> padsBegin5D_Full = {{0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {1, 1, 1, 1, 0}, {2, 0, 1, 0, 1}, {0, 2, 1, 3, 1}};
const std::vector<std::vector<int64_t>> padsEnd5D_Full   = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {1, 0, 1, 1, 2}, {2, 2, 0, 1, 0}, {1, 1, 2, 0, 1}};

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic5DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_GPUPadDynamic5D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Smoke),
                ::testing::ValuesIn(padsEnd5D_Smoke),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        GPUPadDynamic5DConst,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Full),
                ::testing::ValuesIn(padsEnd5D_Full),
                ::testing::ValuesIn(argPadValue),
                ::testing::Values(ngraph::helpers::PadMode::CONSTANT)),
        PadLayerGPUTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        GPUPadDynamic5D,
        PadLayerGPUTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputShapesDynamic5D),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(padsBegin5D_Full),
                ::testing::ValuesIn(padsEnd5D_Full),
                ::testing::Values(0),
                ::testing::ValuesIn(padMode)),
        PadLayerGPUTest::getTestCaseName
);

/* *======================* *=====================* *======================* */

} // namespace
} // namespace GPULayerTestsDefinitions
