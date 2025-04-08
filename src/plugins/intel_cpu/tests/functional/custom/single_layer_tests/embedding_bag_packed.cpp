// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/embedding_bag_packed.hpp"
#include "openvino/op/util/embeddingbag_packed_base.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

typedef std::tuple<InputShape,                                      // input_shapes
                   std::vector<std::vector<size_t>>,                // indices
                   bool,                                            // with_weights
                   ov::op::util::EmbeddingBagPackedBase::Reduction  // reduction
                   >
    embeddingBagPackedParams;

typedef std::tuple<embeddingBagPackedParams,
                   ElementType,  // embedding table
                   ElementType,  // indices
                   ov::test::TargetDevice>
    embeddingBagPackedLayerTestParamsSet;

class EmbeddingBagPackedLayerCPUTest : public testing::WithParamInterface<embeddingBagPackedLayerTestParamsSet>,
                                       virtual public SubgraphBaseTest,
                                       public CPUTestsBase {
public:
    using Reduction = ov::op::util::EmbeddingBagPackedBase::Reduction;
    static std::string getTestCaseName(const testing::TestParamInfo<embeddingBagPackedLayerTestParamsSet>& obj) {
        embeddingBagPackedParams params;
        ElementType netPrecision, indPrecision;
        std::string targetDevice;
        std::tie(params, netPrecision, indPrecision, targetDevice) = obj.param;

        InputShape inputShapes;
        std::vector<std::vector<size_t>> indices;
        bool withWeights;
        Reduction reduction;
        std::tie(inputShapes, indices, withWeights, reduction) = params;

        std::ostringstream result;
        result << "IS=" << inputShapes << "_";
        result << "I" << ov::test::utils::vec2str(indices) << "_";
        result << "WW" << withWeights << "_";
        result << "R" << ov::as_string(reduction) << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "indPRC=" << indPrecision << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() override {
        embeddingBagPackedParams embParams;
        ElementType indPrecision;
        std::tie(embParams, inType, indPrecision, targetDevice) = this->GetParam();

        InputShape inputShapes;
        std::vector<std::vector<size_t>> indices;
        bool withWeights;
        Reduction reduction;
        std::tie(inputShapes, indices, withWeights, reduction) = embParams;

        selectedType = makeSelectedTypeStr("ref", inType);
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes({inputShapes});

        auto emb_table_node = std::make_shared<ov::op::v0::Parameter>(inType, inputShapes.first);
        ov::ParameterVector params = {emb_table_node};

        auto embBag = ov::as_type_ptr<ov::op::v15::EmbeddingBagPacked>(
            ov::test::utils::make_embedding_bag_packed(inType,
                                                       indPrecision,
                                                       emb_table_node,
                                                       indices,
                                                       withWeights,
                                                       reduction));
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(embBag)};
        function = std::make_shared<ov::Model>(results, params, "embeddingBagPacked");
    }
};

TEST_P(EmbeddingBagPackedLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "embeddingBagPacked");
}

namespace {

const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::i32, ElementType::u8};

const std::vector<ElementType> indPrecisions = {ElementType::i64, ElementType::i32};

const std::vector<InputShape> input_shapes = {
    // dynamic input shapes
    {// input model dynamic shapes
     {ov::Dimension::dynamic(), ov::Dimension::dynamic()},
     // input tensor shapes
     {{{5, 6}}, {10, 35}}},
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

const std::vector<std::vector<std::vector<size_t>>> indices = {{{0, 1}, {2, 2}, {3, 4}},
                                                               {{4, 4, 3}, {1, 0, 2}},
                                                               {{1, 2, 1, 2}, {1, 2, 1, 2}}};

const std::vector<ov::op::util::EmbeddingBagPackedBase::Reduction> reduction = {
    ov::op::util::EmbeddingBagPackedBase::Reduction::SUM,
    ov::op::util::EmbeddingBagPackedBase::Reduction::MEAN};

const auto embBagPackedArgSetWthWeights = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                                   ::testing::ValuesIn(indices),
                                                   ::testing::Values(true),
                                                   ::testing::Values(ov::op::util::EmbeddingBagPackedBase::Reduction::SUM));

const auto embBagPackedArgSetNoWeights = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                                   ::testing::ValuesIn(indices),
                                                   ::testing::Values(false),
                                                   ::testing::ValuesIn(reduction));

INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingBagPacked_With_Weights,
                         EmbeddingBagPackedLayerCPUTest,
                         ::testing::Combine(embBagPackedArgSetWthWeights,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(indPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EmbeddingBagPackedLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingBagPacked_No_Weights,
                         EmbeddingBagPackedLayerCPUTest,
                         ::testing::Combine(embBagPackedArgSetNoWeights,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(indPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EmbeddingBagPackedLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
