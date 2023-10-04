// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

#include "shared_test_classes/single_layer/pooling.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ov::test;

namespace GPULayerTestsDefinitions {

using poolLayerGpuTestParamsSet = std::tuple<LayerTestsDefinitions::poolSpecificParams,
                                             InputShape,
                                             ElementType>;

class PoolingLayerGPUTest : public testing::WithParamInterface<poolLayerGpuTestParamsSet>,
                            virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<poolLayerGpuTestParamsSet>& obj) {
        LayerTestsDefinitions::poolSpecificParams basicParamsSet;
        InputShape inputShapes;
        ElementType inPrc;
        std::tie(basicParamsSet, inputShapes, inPrc) = obj.param;

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> kernel, stride;
        std::vector<size_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::op::RoundingType roundingType;
        bool excludePad;
        std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

        std::ostringstream results;
        results << "IS=(";
        results << ov::test::utils::partialShape2str({inputShapes.first}) << ")_";
        results << "TS=";
        for (const auto& shape : inputShapes.second) {
            results << ov::test::utils::vec2str(shape) << "_";
        }
        results << "Prc=" << inPrc << "_";
        switch (poolType) {
            case ngraph::helpers::PoolingTypes::MAX:
                results << "MaxPool_";
                break;
            case ngraph::helpers::PoolingTypes::AVG:
                results << "AvgPool_";
                results << "ExcludePad=" << excludePad << "_";
                break;
        }
        results << "K" << ov::test::utils::vec2str(kernel) << "_";
        results << "S" << ov::test::utils::vec2str(stride) << "_";
        results << "PB" << ov::test::utils::vec2str(padBegin) << "_";
        results << "PE" << ov::test::utils::vec2str(padEnd) << "_";
        results << "Rounding=" << roundingType << "_";
        results << "AutoPad=" << padType << "_";

        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        LayerTestsDefinitions::poolSpecificParams basicParamsSet;
        InputShape inputShapes;
        ElementType inPrc;
        std::tie(basicParamsSet, inputShapes, inPrc) = this->GetParam();

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> kernel, stride;
        std::vector<size_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::op::RoundingType roundingType;
        bool excludePad;
        std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = basicParamsSet;

        init_input_shapes({inputShapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inPrc, shape));
        }
        std::shared_ptr<ngraph::Node> poolInput = params[0];

        std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(poolInput,
                                                                             stride,
                                                                             padBegin,
                                                                             padEnd,
                                                                             kernel,
                                                                             roundingType,
                                                                             padType,
                                                                             excludePad,
                                                                             poolType);

        auto makeFunction = [](const ngraph::element::Type &ngPrc, ngraph::ParameterVector &params, const std::shared_ptr<ngraph::Node> &lastNode) {
            ngraph::ResultVector results;

            for (size_t i = 0; i < lastNode->get_output_size(); i++)
                results.push_back(std::make_shared<ngraph::opset1::Result>(lastNode->output(i)));

            return std::make_shared<ngraph::Function>(results, params, "PoolingGPU");
        };
        function = makeFunction(inPrc, params, pooling);
    }
};

TEST_P(PoolingLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
}

namespace {

const std::vector<ElementType> inpOutPrecision = { ElementType::f32 };

const std::vector<InputShape> inputShapes3D = {
        { {}, {{3, 4, 64}} },
        { {}, {{2, 8, 12}} },
        { {}, {{1, 16, 12}} },
        { {}, {{1, 21, 4}} },
        { {}, {{1, 32, 8}} },
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {1, 32, 8},
                {1, 21, 4},
                {2, 8, 12}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}},
            // target
            {
                {3, 4, 64},
                {1, 16, 12},
                {1, 32, 8}
            }
        }
};

const std::vector<InputShape> inputShapes4D = {
        { {}, {{3, 4, 64, 64}} },
        { {}, {{2, 8, 8, 12}} },
        { {}, {{1, 16, 16, 12}} },
        { {}, {{1, 21, 8, 4}} },
        { {}, {{1, 32, 8, 8}} },
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 32, 8, 8},
                {1, 21, 8, 4},
                {2, 8, 8, 12}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}, {1, 64}},
            // target
            {
                {3, 4, 64, 64},
                {1, 16, 16, 12},
                {1, 32, 8, 8}
            }
        },
        {
            // dynamic
            {{1, 10}, 16, 8, 8},
            // target
            {
                {1, 16, 8, 8},
                {2, 16, 8, 8},
            }
        }
};

const std::vector<InputShape> inputShapes4D_Large = {
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 16, 65, 65},
                {1, 8, 130, 130},
                {1, 16, 65, 65}
            }
        },
};

const std::vector<InputShape> inputShapes5D = {
        { {}, {{1, 4, 16, 16, 16}} },
        { {}, {{2, 8, 8, 8, 8}} },
        { {}, {{2, 16, 12, 16, 20}} },
        { {}, {{1, 19, 16, 20, 8}} },
        { {}, {{1, 32, 16, 8, 12}} },
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {2, 8, 8, 8, 8},
                {1, 19, 16, 20, 8},
                {1, 4, 16, 16, 16}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}, {1, 64}, {1, 25}},
            // target
            {
                {1, 4, 16, 16, 16},
                {1, 32, 16, 8, 12},
                {3, 16, 4, 8, 3}
            }
        }
};

/* ============= Pooling (1D) ============= */
const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax3D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2}, {2}, {0}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4}, {2}, {0}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2}, {1}, {0}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg3D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4}, {4}, {2}, {2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_GPU_3D, PoolingLayerGPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMax3D),
                                 ::testing::ValuesIn(inputShapes3D),
                                 ::testing::ValuesIn(inpOutPrecision)),
                         PoolingLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_GPU_3D, PoolingLayerGPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D),
                                 ::testing::ValuesIn(inputShapes3D),
                                 ::testing::ValuesIn(inpOutPrecision)),
                         PoolingLayerGPUTest::getTestCaseName);

/* ============= Pooling (2D) ============= */
const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax4D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 1}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true }
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_GPU_4D, PoolingLayerGPUTest,
                            ::testing::Combine(
                            ::testing::ValuesIn(paramsMax4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision)),
                        PoolingLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_GPU_4D, PoolingLayerGPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision)),
                        PoolingLayerGPUTest::getTestCaseName);

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D_Large = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {65, 65}, {65, 65}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID, true },
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_GPU_Large, PoolingLayerGPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_Large),
                            ::testing::ValuesIn(inputShapes4D_Large),
                            ::testing::ValuesIn(inpOutPrecision)),
                        PoolingLayerGPUTest::getTestCaseName);

/* ============= Pooling (3D) ============= */
const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax5D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4, 4}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true }
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_GPU_5D, PoolingLayerGPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(paramsMax5D),
                             ::testing::ValuesIn(inputShapes5D),
                             ::testing::ValuesIn(inpOutPrecision)),
                         PoolingLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_GPU_5D, PoolingLayerGPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D),
                              ::testing::ValuesIn(inputShapes5D),
                              ::testing::ValuesIn(inpOutPrecision)),
                          PoolingLayerGPUTest::getTestCaseName);
} // namespace

} // namespace GPULayerTestsDefinitions
