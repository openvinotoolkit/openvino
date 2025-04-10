// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using GatherNDLayerCPUTestParamSet = std::tuple<InputShape,                          // Input shapes
                                                std::pair<Shape, std::vector<int>>,  // Indexes shape and values
                                                ElementType,                         // Input element type
                                                ElementType,                         // Indices element type
                                                int                                  // Batch dims
                                                >;

class GatherNDLayerCPUTest : public testing::WithParamInterface<GatherNDLayerCPUTestParamSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherNDLayerCPUTestParamSet> obj) {
        InputShape shapes;
        std::pair<Shape, std::vector<int>> indexes;
        ElementType dataElementType, idxElementType;
        int batchDims;
        std::tie(shapes, indexes, dataElementType, idxElementType, batchDims) = obj.param;

        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& item : shapes.second) {
            results << ov::test::utils::vec2str(item) << "_";
        }
        results << "IDXShape=" << ov::test::utils::vec2str(indexes.first) << "_";
        results << "SRCPrc=" << dataElementType << "_";
        results << "IDXPrc=" << idxElementType << "_";
        results << "BD=" << batchDims << "_";

        return results.str();
    }

protected:
    void SetUp() override {
        InputShape shapes;
        std::pair<Shape, std::vector<int>> indexes;
        ElementType dataElementType, idxElementType;
        int batchDims;
        std::tie(shapes, indexes, dataElementType, idxElementType, batchDims) = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(dataElementType, shape));
        }
        auto indexes_node = ov::op::v0::Constant::create(idxElementType, indexes.first, indexes.second);
        auto gather_nd = std::make_shared<ov::op::v5::GatherND>(params[0], indexes_node, batchDims);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_nd)};
        function = std::make_shared<ov::Model>(results, params, "gatherND");
    }
};

class GatherND8LayerCPUTest : public testing::WithParamInterface<GatherNDLayerCPUTestParamSet>,
                              virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherNDLayerCPUTestParamSet> obj) {
        return GatherNDLayerCPUTest::getTestCaseName(obj);
    }

protected:
    void SetUp() override {
        InputShape shapes;
        std::pair<Shape, std::vector<int>> indexes;
        ElementType dataElementType, idxElementType;
        int batchDims;
        std::tie(shapes, indexes, dataElementType, idxElementType, batchDims) = this->GetParam();

        targetDevice = ov::test::utils::DEVICE_CPU;
        init_input_shapes({shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(dataElementType, shape));
        }
        auto indexes_node = ov::op::v0::Constant::create(idxElementType, indexes.first, indexes.second);
        auto gather_nd = std::make_shared<ov::op::v8::GatherND>(params[0], indexes_node, batchDims);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_nd)};
        function = std::make_shared<ov::Model>(results, params, "gatherND");
    }
};

TEST_P(GatherNDLayerCPUTest, CompareWithRefs) {
    run();
}

TEST_P(GatherND8LayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<ElementType> inputPrecisions = {ElementType::f32, ElementType::bf16, ElementType::i8};

const std::vector<ElementType> indexesPrecisions = {ElementType::i32};

const std::vector<InputShape> inputShapesDynamicBD_0 = {
    {{-1, -1, -1},                                      // dynamic
     {{5, 10, 5}, {4, 12, 4}, {4, 12, 4}, {5, 5, 5}}},  // target

    {{-1, 5, -1, -1},                              // dynamic
     {{8, 5, 5, 5}, {5, 5, 8, 4}, {4, 5, 4, 5}}},  // target

    {{{4, 10}, {5, 10}, {5, 10}, {5, 10}, {5, 10}},          // dynamic
     {{4, 5, 5, 5, 5}, {4, 5, 5, 8, 5}, {10, 8, 5, 5, 5}}},  // target
};

const std::vector<std::pair<Shape, std::vector<int>>> indexesShapesBD_0 = {
    std::pair<Shape, std::vector<int>>{{2, 2}, {3, 3, 2, 1}},
    std::pair<Shape, std::vector<int>>{{1, 2, 3}, {0, 1, 1, 1, 0, 2}},
    std::pair<Shape, std::vector<int>>{{2, 1, 1, 2}, {0, 2, 1, 1}},
};

const auto subset_BD0 = ::testing::Combine(::testing::ValuesIn(inputShapesDynamicBD_0),
                                           ::testing::ValuesIn(indexesShapesBD_0),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::ValuesIn(indexesPrecisions),
                                           ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5DynamicBD_0,
                         GatherNDLayerCPUTest,
                         subset_BD0,
                         GatherNDLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_GatherND8DynamicBD_0,
                         GatherND8LayerCPUTest,
                         subset_BD0,
                         GatherNDLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapesDynamicBD_1 = {
    {{3, -1, -1},                                       // dynamic
     {{3, 10, 5}, {3, 10, 5}, {3, 12, 8}, {3, 8, 8}}},  // target

    {{3, {5, 10}, {5, 10}, {5, 10}, {5, 10}},                  // dynamic
     {{3, 5, 5, 5, 5}, {3, 8, 10, 10, 10}, {3, 8, 6, 8, 7}}},  // target
};

const std::vector<std::pair<Shape, std::vector<int>>> indexesShapesBD_1 = {
    std::pair<Shape, std::vector<int>>{{3, 2}, {0, 1, 2, 1, 0, 0}},
    std::pair<Shape, std::vector<int>>{{3, 2, 2}, {0, 1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 2}},
    std::pair<Shape, std::vector<int>>{{3, 1, 1, 2}, {0, 2, 1, 1, 0, 2}},
};

const auto subset_BD1 = ::testing::Combine(::testing::ValuesIn(inputShapesDynamicBD_1),
                                           ::testing::ValuesIn(indexesShapesBD_1),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::ValuesIn(indexesPrecisions),
                                           ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5DynamicBD_1,
                         GatherNDLayerCPUTest,
                         subset_BD1,
                         GatherNDLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_GatherND8DynamicBD_1,
                         GatherND8LayerCPUTest,
                         subset_BD1,
                         GatherNDLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapesDynamicBD_2 = {
    {{2, 2, -1, -1, -1},                                                     // dynamic
     {{2, 2, 5, 6, 5}, {2, 2, 2, 3, 3}, {2, 2, 2, 3, 3}, {2, 2, 7, 2, 3}}},  // target

    {{2, 2, {5, 10}, {5, 10}, {5, 10}},                       // dynamic
     {{2, 2, 5, 5, 5}, {2, 2, 10, 10, 5}, {2, 2, 7, 8, 7}}},  // target
};

const std::vector<std::pair<Shape, std::vector<int>>> indexesShapesBD_2 = {
    std::pair<Shape, std::vector<int>>{{2, 2, 3}, {0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0}},
    std::pair<Shape, std::vector<int>>{{2, 2, 2, 3},
                                       {0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0}},
};

const auto subset_BD2 = ::testing::Combine(::testing::ValuesIn(inputShapesDynamicBD_2),
                                           ::testing::ValuesIn(indexesShapesBD_2),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::ValuesIn(indexesPrecisions),
                                           ::testing::Values(0));

INSTANTIATE_TEST_SUITE_P(smoke_GatherND5DynamicBD_2,
                         GatherNDLayerCPUTest,
                         subset_BD2,
                         GatherNDLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_GatherND8DynamicBD_2,
                         GatherND8LayerCPUTest,
                         subset_BD2,
                         GatherNDLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
