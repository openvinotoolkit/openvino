// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/embedding_bag_offsets.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

typedef std::tuple<InputShape,                                      // input_shapes
                   std::vector<size_t>,                             // indices
                   std::vector<size_t>,                             // offsets
                   size_t,                                          // default_index
                   bool,                                            // with_weights
                   bool,                                            // with_def_index
                   ov::op::util::EmbeddingBagOffsetsBase::Reduction // reduction
                   >
    embeddingBagOffsetsParams;

typedef std::tuple<embeddingBagOffsetsParams,
                   ElementType,  // embedding table
                   ElementType,  // indices
                   ov::test::TargetDevice>
    embeddingBagOffsetsLayerTestParamsSet;

class EmbeddingBagOffsetsLayerCPUTest : public testing::WithParamInterface<embeddingBagOffsetsLayerTestParamsSet>,
                                        virtual public SubgraphBaseTest,
                                        public CPUTestsBase {
public:
    using Reduction = ov::op::util::EmbeddingBagOffsetsBase::Reduction;
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagOffsetsLayerTestParamsSet>& obj) {
        embeddingBagOffsetsParams params;
        ElementType netPrecision, indPrecision;
        std::string targetDevice;
        std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;

        InputShape inputShapes;
        std::vector<size_t> indices, offsets;
        size_t defaultIndex;
        bool withWeights, withDefIndex;
        Reduction reduction;
        std::tie(inputShapes, indices, offsets, defaultIndex, withWeights, withDefIndex, reduction) = params;

        std::ostringstream result;
        result << "IS=" << inputShapes << "_";
        result << "I" << ov::test::utils::vec2str(indices) << "_";
        result << "O" << ov::test::utils::vec2str(offsets) << "_";
        result << "DI" << defaultIndex << "_";
        result << "WW" << withWeights << "_";
        result << "WDI" << withDefIndex << "_";
        result << "R" << ov::as_string(reduction) << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "indPRC=" << indPrecision << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void SetUp() override {
        embeddingBagOffsetsParams embParams;
        ElementType indPrecision;
        std::tie(embParams, inType, indPrecision, targetDevice) = this->GetParam();

        InputShape inputShapes;
        std::vector<size_t> indices, offsets;
        bool withWeights, withDefIndex;
        Reduction reduction;
        size_t defaultIndex;
        std::tie(inputShapes, indices, offsets, defaultIndex, withWeights, withDefIndex, reduction) = embParams;

        selectedType = makeSelectedTypeStr("ref", inType);
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes({inputShapes});

        auto emb_table_node = std::make_shared<ov::op::v0::Parameter>(inType, inputShapes.first);
        ov::ParameterVector params = {emb_table_node};

        auto embBag = ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(
            ov::test::utils::make_embedding_bag_offsets(inType,
                                                        indPrecision,
                                                        emb_table_node,
                                                        indices,
                                                        offsets,
                                                        defaultIndex,
                                                        withWeights,
                                                        withDefIndex,
                                                        reduction));
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(embBag)};
        function = std::make_shared<ov::Model>(results, params, "embeddingBagOffsets");
    }
};

TEST_P(EmbeddingBagOffsetsLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "embeddingBagOffsets");
}

namespace {

const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::i32, ElementType::u8};

const std::vector<ElementType> indPrecisions = {ElementType::i64, ElementType::i32};

const std::vector<InputShape> input_shapes = {
    // dynamic input shapes
    {// input model dynamic shapes
     {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
     // input tensor shapes
     {{5, 6}, {10, 35}}},
    {// input model dynamic shapes
     {ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
     // input tensor shapes
     {{5, 4, 16}, {10, 12, 8}}},
    {// input model dynamic shapes with limits
     {{5, 10}, {6, 35}, {4, 8}},
     // input tensor shapes
     {{5, 6, 4}, {10, 35, 8}, {5, 6, 4}}},
    // static shapes
    {{5, 6}, {{5, 6}}},
    {{10, 35}, {{10, 35}}},
    {{5, 4, 16}, {{5, 4, 16}}},
};

const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}, {4, 4, 3, 1, 0}, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2}};
const std::vector<std::vector<size_t>> offsets = {{0, 2}, {0, 0, 2, 2}, {2, 4}};
const std::vector<size_t> default_index = {0, 4};
const std::vector<bool> with_default_index = {false, true};
const std::vector<ov::op::util::EmbeddingBagOffsetsBase::Reduction> reduction = {
    ov::op::util::EmbeddingBagOffsetsBase::Reduction::SUM,
    ov::op::util::EmbeddingBagOffsetsBase::Reduction::MEAN};

const auto embBagOffsetArgSetWthWeights = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                                   ::testing::ValuesIn(indices),
                                                   ::testing::ValuesIn(offsets),
                                                   ::testing::ValuesIn(default_index),
                                                   ::testing::Values(true),
                                                   ::testing::ValuesIn(with_default_index),
                                                   ::testing::Values(ov::op::util::EmbeddingBagOffsetsBase::Reduction::SUM));
const auto embBagOffsetArgSetNoWeights = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                                   ::testing::ValuesIn(indices),
                                                   ::testing::ValuesIn(offsets),
                                                   ::testing::ValuesIn(default_index),
                                                   ::testing::Values(false),
                                                   ::testing::ValuesIn(with_default_index),
                                                   ::testing::ValuesIn(reduction));
INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingBagOffsets_With_Weights,
                         EmbeddingBagOffsetsLayerCPUTest,
                         ::testing::Combine(embBagOffsetArgSetWthWeights,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(indPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EmbeddingBagOffsetsLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingBagOffsets_No_Weights,
                         EmbeddingBagOffsetsLayerCPUTest,
                         ::testing::Combine(embBagOffsetArgSetNoWeights,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(indPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EmbeddingBagOffsetsLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
