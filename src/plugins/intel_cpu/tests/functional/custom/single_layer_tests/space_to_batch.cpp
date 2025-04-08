// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {
std::vector<int64_t> blockShape, padsBegin, padsEnd;
ov::Shape paramShape;
}  // namespace

using SpaceToBatchLayerTestCPUParams = std::tuple<std::vector<InputShape>,  // Input shapes
                                                  std::vector<int64_t>,     // block shape
                                                  std::vector<int64_t>,     // pads begin
                                                  std::vector<int64_t>,     // pads end
                                                  ov::element::Type,        // Network type
                                                  CPUSpecificParams>;

class SpaceToBatchCPULayerTest : public testing::WithParamInterface<SpaceToBatchLayerTestCPUParams>,
                                 virtual public SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SpaceToBatchLayerTestCPUParams>& obj) {
        std::vector<InputShape> inputShapes;
        ov::element::Type netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, padsBegin, padsEnd, netPrecision, cpuParams) = obj.param;
        std::ostringstream result;
        if (inputShapes.front().first.size() != 0) {
            result << "IS=(";
            for (const auto& shape : inputShapes) {
                result << ov::test::utils::partialShape2str({shape.first}) << "_";
            }
            result.seekp(-1, result.cur);
            result << ")_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "blockShape=" << ov::test::utils::vec2str(blockShape) << "_";
        result << "padsBegin=" << ov::test::utils::vec2str(padsBegin) << "_";
        result << "padsEnd=" << ov::test::utils::vec2str(padsEnd) << "_";
        result << "netPRC=" << netPrecision.get_type_name() << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0U) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 2560;
                in_data.resolution = 256;
                tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else if (i == 1U) {
                tensor = ov::Tensor(funcInput.get_element_type(), paramShape);
                auto* dataPtr = tensor.data<int64_t>();
                for (size_t j = 0; j < blockShape.size(); j++) {
                    dataPtr[j] = blockShape[j];
                }
            } else if (i == 2U) {
                tensor = ov::Tensor(funcInput.get_element_type(), paramShape);
                auto* dataPtr = tensor.data<int64_t>();
                for (size_t j = 0; j < padsBegin.size(); j++) {
                    dataPtr[j] = padsBegin[j];
                }
            } else if (i == 3U) {
                tensor = ov::Tensor(funcInput.get_element_type(), paramShape);
                auto* dataPtr = tensor.data<int64_t>();
                for (size_t j = 0; j < padsEnd.size(); j++) {
                    dataPtr[j] = padsEnd[j];
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::vector<InputShape> inputShapes;
        ov::element::Type netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, padsBegin, padsEnd, netPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        const std::vector<InputShape> inputShapesVec{inputShapes};
        init_input_shapes(inputShapesVec);

        if (strcmp(netPrecision.get_type_name().c_str(), "u8") == 0)
            selectedType = std::string("ref_any_") + "i8";
        else
            selectedType = std::string("ref_any_") + netPrecision.get_type_name();

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes.front())};
        paramShape = {params[0]->get_partial_shape().size()};

        std::shared_ptr<ov::Node> in2, in3, in4;
        auto blockShapeParam = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, paramShape);
        in2 = blockShapeParam;
        params.push_back(blockShapeParam);
        auto padsBeginParam = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, paramShape);
        in3 = padsBeginParam;
        params.push_back(padsBeginParam);
        auto padsEndParam = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, paramShape);
        in4 = padsEndParam;
        params.push_back(padsEndParam);

        auto s2b = std::make_shared<ov::op::v1::SpaceToBatch>(params[0], in2, in3, in4);
        function = makeNgraphFunction(inType, params, s2b, "SpaceToBatchCPU");
    }
};

TEST_P(SpaceToBatchCPULayerTest, CompareWithRefs) {
    run();
    CPUTestsBase::CheckPluginRelatedResults(compiledModel, "SpaceToBatch");
};

namespace {

const std::vector<ov::element::Type> netPrecision = {ov::element::u8,
                                                     ov::element::i8,
                                                     ov::element::i32,
                                                     ov::element::f32,
                                                     ov::element::bf16};

const std::vector<std::vector<int64_t>> blockShape4D1 = {{1, 2, 1, 2}, {1, 1, 2, 2}, {1, 2, 2, 2}};
const std::vector<std::vector<int64_t>> padsBegin4D1 = {{0, 0, 0, 1}, {0, 0, 2, 1}, {0, 0, 4, 3}};
const std::vector<std::vector<int64_t>> padsEnd4D1 = {{0, 0, 0, 1}, {0, 0, 4, 1}, {0, 0, 2, 3}};

std::vector<std::vector<ov::Shape>> staticInputShapes4D1 = {
    {{1, 16, 8, 12}, {4}, {4}, {4}},
    {{1, 32, 8, 8}, {4}, {4}, {4}},
};

std::vector<std::vector<InputShape>> dynamicInputShapes4D1 = {
    {{{-1, -1, -1, -1}, {{1, 6, 4, 8}, {2, 4, 8, 10}, {1, 8, 4, 10}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}},
    {{{{1, 4}, {2, 16}, 6, -1}, {{4, 8, 6, 4}, {1, 6, 6, 8}, {2, 12, 6, 4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes4D1Blocked = {
    {{{-1, 16, -1, -1}, {{1, 16, 4, 6}, {2, 16, 6, 6}, {4, 16, 4, 8}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

const std::vector<std::vector<int64_t>> blockShape4D2 = {{1, 2, 4, 3}, {1, 4, 4, 1}};
const std::vector<std::vector<int64_t>> padsBegin4D2 = {{0, 0, 0, 0}, {0, 0, 4, 3}};
const std::vector<std::vector<int64_t>> padsEnd4D2 = {{0, 0, 4, 0}, {0, 0, 4, 3}};

std::vector<std::vector<ov::Shape>> staticInputShapes4D2 = {
    {{1, 16, 12, 12}, {4}, {4}, {4}},
    {{1, 32, 12, 15}, {4}, {4}, {4}},
};

std::vector<std::vector<InputShape>> dynamicInputShapes4D2 = {
    {{{-1, -1, -1, -1}, {{1, 4, 8, 9}, {2, 8, 12, 9}, {6, 12, 4, 12}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}},
    {{{2, {4, 16}, -1, -1}, {{2, 8, 4, 9}, {2, 4, 8, 6}, {2, 12, 12, 3}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes4D2Blocked = {
    {{{-1, 16, -1, -1}, {{2, 16, 4, 15}, {2, 16, 8, 12}, {3, 16, 12, 9}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

const std::vector<CPUSpecificParams> cpuParamsWithBlock_4D = {CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
                                                              CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}),
                                                              CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
                                                              CPUSpecificParams({nchw}, {nchw}, {}, {})};

const std::vector<CPUSpecificParams> cpuParams_4D = {CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
                                                     CPUSpecificParams({nchw}, {nchw}, {}, {})};

const auto staticSpaceToBatchParamsSet4D1 =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D1)),
                       ::testing::ValuesIn(blockShape4D1),
                       ::testing::ValuesIn(padsBegin4D1),
                       ::testing::ValuesIn(padsEnd4D1),
                       ::testing::ValuesIn(netPrecision),
                       ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsSet4D1 = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D1),
                                                                ::testing::ValuesIn(blockShape4D1),
                                                                ::testing::ValuesIn(padsBegin4D1),
                                                                ::testing::ValuesIn(padsEnd4D1),
                                                                ::testing::ValuesIn(netPrecision),
                                                                ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsWithBlockedSet4D1 =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D1Blocked),
                       ::testing::ValuesIn(blockShape4D1),
                       ::testing::ValuesIn(padsBegin4D1),
                       ::testing::ValuesIn(padsEnd4D1),
                       ::testing::ValuesIn(netPrecision),
                       ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto staticSpaceToBatchParamsSet4D2 =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D2)),
                       ::testing::ValuesIn(blockShape4D2),
                       ::testing::ValuesIn(padsBegin4D2),
                       ::testing::ValuesIn(padsEnd4D2),
                       ::testing::ValuesIn(netPrecision),
                       ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsSet4D2 = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D2),
                                                                ::testing::ValuesIn(blockShape4D2),
                                                                ::testing::ValuesIn(padsBegin4D2),
                                                                ::testing::ValuesIn(padsEnd4D2),
                                                                ::testing::ValuesIn(netPrecision),
                                                                ::testing::ValuesIn(cpuParams_4D));

const auto dynamicSpaceToBatchParamsWithBlockedSet4D2 =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D2Blocked),
                       ::testing::ValuesIn(blockShape4D2),
                       ::testing::ValuesIn(padsBegin4D2),
                       ::testing::ValuesIn(padsEnd4D2),
                       ::testing::ValuesIn(netPrecision),
                       ::testing::ValuesIn(cpuParamsWithBlock_4D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticSpaceToBatchCPULayerTestCase1_4D,
                         SpaceToBatchCPULayerTest,
                         staticSpaceToBatchParamsSet4D1,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCase1_4D,
                         SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsSet4D1,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCaseWithBlocked1_4D,
                         SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsWithBlockedSet4D1,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticSpaceToBatchCPULayerTestCase2_4D,
                         SpaceToBatchCPULayerTest,
                         staticSpaceToBatchParamsSet4D2,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCase2_4D,
                         SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsSet4D2,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCaseWithBlocked2_4D,
                         SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsWithBlockedSet4D2,
                         SpaceToBatchCPULayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> staticInputShapes4DPE = {{{1, 2, 9, 1}, {4}, {4}, {4}}};
INSTANTIATE_TEST_SUITE_P(
    smoke_StaticSpaceToBatch_4D_parallel_block_edge,
    SpaceToBatchCPULayerTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4DPE)),
                       ::testing::Values(std::vector<int64_t>{1, 4, 3, 1}),
                       ::testing::Values(std::vector<int64_t>{0, 1, 2, 0}),
                       ::testing::Values(std::vector<int64_t>{0, 1, 4, 0}),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(CPUSpecificParams{})),
    SpaceToBatchCPULayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> blockShape5D = {{1, 1, 2, 2, 1}, {1, 2, 4, 1, 3}};
const std::vector<std::vector<int64_t>> padsBegin5D = {{0, 0, 0, 0, 0}, {0, 0, 4, 0, 0}, {0, 0, 0, 2, 3}};
const std::vector<std::vector<int64_t>> padsEnd5D = {{0, 0, 0, 0, 0}, {0, 0, 0, 4, 3}, {0, 0, 4, 2, 3}};

std::vector<std::vector<ov::Shape>> staticInputShapes5D = {{{2, 16, 4, 6, 12}, {5}, {5}, {5}},
                                                           {{1, 32, 8, 8, 6}, {5}, {5}, {5}},
                                                           {{1, 16, 4, 12, 12}, {5}, {5}, {5}}};

std::vector<std::vector<InputShape>> dynamicInputShapes5D = {
    {{{-1, -1, -1, -1, -1}, {{2, 2, 12, 4, 15}, {4, 4, 8, 6, 9}, {3, 6, 4, 2, 12}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}},
    {{{{1, 10}, {2, 20}, {4, 50}, -1, -1}, {{3, 12, 8, 6, 9}, {5, 10, 4, 8, 15}, {6, 8, 20, 4, 12}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes5DBlocked = {
    {{{-1, 16, -1, -1, -1}, {{2, 16, 4, 6, 9}, {5, 16, 16, 4, 6}, {7, 16, 8, 2, 3}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}}};

const std::vector<CPUSpecificParams> cpuParamsWithBlock_5D = {CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
                                                              CPUSpecificParams({nCdhw8c}, {nCdhw8c}, {}, {}),
                                                              CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
                                                              CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})};

const std::vector<CPUSpecificParams> cpuParams_5D = {CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
                                                     CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})};

const auto staticSpaceToBatchParamsSet5D =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes5D)),
                       ::testing::ValuesIn(blockShape5D),
                       ::testing::ValuesIn(padsBegin5D),
                       ::testing::ValuesIn(padsEnd5D),
                       ::testing::ValuesIn(netPrecision),
                       ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto dynamicSpaceToBatchParamsSet5D = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D),
                                                               ::testing::ValuesIn(blockShape5D),
                                                               ::testing::ValuesIn(padsBegin5D),
                                                               ::testing::ValuesIn(padsEnd5D),
                                                               ::testing::ValuesIn(netPrecision),
                                                               ::testing::ValuesIn(cpuParams_5D));

const auto dynamicSpaceToBatchParamsWithBlockedSet5D =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes5DBlocked),
                       ::testing::ValuesIn(blockShape5D),
                       ::testing::ValuesIn(padsBegin5D),
                       ::testing::ValuesIn(padsEnd5D),
                       ::testing::ValuesIn(netPrecision),
                       ::testing::ValuesIn(cpuParamsWithBlock_5D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticSpaceToBatchCPULayerTestCase_5D,
                         SpaceToBatchCPULayerTest,
                         staticSpaceToBatchParamsSet5D,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCase_5D,
                         SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsSet5D,
                         SpaceToBatchCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicSpaceToBatchCPULayerTestCaseWithBlocked_5D,
                         SpaceToBatchCPULayerTest,
                         dynamicSpaceToBatchParamsWithBlockedSet5D,
                         SpaceToBatchCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
