// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <sstream>
#include <vector>

#include <openvino/core/partial_shape.hpp>
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
    InputShape, // input_shapes
    std::vector<size_t>, // indices
    std::vector<size_t>, // segment_ids
    size_t,              // num_segments
    size_t,              // default_index
    bool,                // with_weights
    bool                 // with_def_index
    > embeddingSegmentsSumParams;

typedef std::tuple<
    embeddingSegmentsSumParams,
    ElementType, // embedding table
    ElementType, // indices
    LayerTestsUtils::TargetDevice> embeddingSegmentsSumLayerTestParamsSet;

class EmbeddingSegmentsSumLayerCPUTest :
        public testing::WithParamInterface<embeddingSegmentsSumLayerTestParamsSet>,
        virtual public SubgraphBaseTest,
        public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingSegmentsSumLayerTestParamsSet>& obj) {
        embeddingSegmentsSumParams params;
        ElementType netPrecision, indPrecision;
        std::string targetDevice;
        std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;

        InputShape inputShapes;
        std::vector<size_t> indices, segmentIds;
        size_t numSegments, defaultIndex;
        bool withWeights, withDefIndex;
        std::tie(inputShapes, indices, segmentIds, numSegments, defaultIndex, withWeights, withDefIndex) = params;

        std::ostringstream result;
        result << "IS=" << inputShapes << "_";
        result << "I"  << ov::test::utils::vec2str(indices) << "_";
        result << "SI" << ov::test::utils::vec2str(segmentIds) << "_";
        result << "NS" << numSegments << "_";
        result << "DI" << defaultIndex << "_";
        result << "WW" << withWeights << "_";
        result << "WDI" << withDefIndex << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "indPRC=" << indPrecision << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        embeddingSegmentsSumParams embParams;
        ElementType indPrecision;
        std::tie(embParams, inType, indPrecision, targetDevice) = this->GetParam();

        InputShape inputShapes;
        std::vector<size_t> indices, segmentIds;
        bool withWeights, withDefIndex;
        size_t numSegments, defaultIndex;
        std::tie(inputShapes, indices, segmentIds, numSegments, defaultIndex, withWeights, withDefIndex) = embParams;

        selectedType = makeSelectedTypeStr("ref", inType);
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes({ inputShapes });

        auto emb_table_node = std::make_shared<ngraph::opset1::Parameter>(inType, inputShapes.first);
        ngraph::ParameterVector params = {emb_table_node};

        auto embBag = std::dynamic_pointer_cast<ngraph::opset3::EmbeddingSegmentsSum>(ngraph::builder::makeEmbeddingSegmentsSum(
            inType,
            indPrecision,
            emb_table_node,
            indices,
            segmentIds,
            numSegments,
            defaultIndex,
            withWeights,
            withDefIndex));
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(embBag)};
        function = std::make_shared<ngraph::Function>(results, params, "embeddingSegmentsSum");
    }
};

TEST_P(EmbeddingSegmentsSumLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "embeddingSegmentsSum");
}

namespace {
const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::i32,
        ElementType::u8
};

const std::vector<ElementType> indPrecisions = {
        ElementType::i64,
        ElementType::i32
};

const std::vector<InputShape> input_shapes = {
    // dynamic input shapes
    {
        // input model dynamic shapes
        {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        // input tensor shapes
        {{5, 6}, {10, 35}}
    },
    {
        // input model dynamic shapes
        {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
        // input tensor shapes
        {{5, 4, 16}, {10, 12, 8}}
    },
    {
        // input model dynamic shapes with limits
        {{5, 10}, {6, 35}, {4, 8}},
        // input tensor shapes
        {{5, 6, 4}, {10, 35, 8}, {5, 6, 4}}
    },
    // static shapes
    {{5, 6}, {{5, 6}}},
    {{10, 35}, {{10, 35}}},
    {{5, 4, 16}, {{5, 4, 16}}},
};

const std::vector<std::vector<size_t>> indices =
    {{0, 1, 2, 2, 3}, {4, 4, 3, 1, 2}};
const std::vector<std::vector<size_t>> segment_ids = {{0, 1, 2, 3, 4}, {0, 0, 2, 2, 4}};
const std::vector<size_t> num_segments = {5, 7};
const std::vector<size_t> default_index = {0, 4};
const std::vector<bool> with_weights = {false, true};
const std::vector<bool> with_default_index = {false, true};

const auto embSegmentsSumArgSet = ::testing::Combine(
    ::testing::ValuesIn(input_shapes),
    ::testing::ValuesIn(indices),
    ::testing::ValuesIn(segment_ids),
    ::testing::ValuesIn(num_segments),
    ::testing::ValuesIn(default_index),
    ::testing::ValuesIn(with_weights),
    ::testing::ValuesIn(with_default_index)
);

INSTANTIATE_TEST_SUITE_P(smoke, EmbeddingSegmentsSumLayerCPUTest,
     ::testing::Combine(
         embSegmentsSumArgSet,
         ::testing::ValuesIn(netPrecisions),
         ::testing::ValuesIn(indPrecisions),
         ::testing::Values(ov::test::utils::DEVICE_CPU)),
         EmbeddingSegmentsSumLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace CPULayerTestsDefinitions
