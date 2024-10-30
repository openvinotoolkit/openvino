// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

namespace {
std::vector<int64_t> blockShape, cropsBegin, cropsEnd;
}  // namespace

using BatchToSpaceLayerTestCPUParams = std::tuple<std::vector<InputShape>,  // Input shapes
                                                  std::vector<int64_t>,     // block shape
                                                  std::vector<int64_t>,     // crops begin
                                                  std::vector<int64_t>,     // crops end
                                                  ov::element::Type,        // Network precision
                                                  CPUSpecificParams>;

class BatchToSpaceCPULayerTest : public testing::WithParamInterface<BatchToSpaceLayerTestCPUParams>,
                                 virtual public SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchToSpaceLayerTestCPUParams>& obj) {
        std::vector<InputShape> inputShapes;
        ov::element::Type model_type;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, cropsBegin, cropsEnd, model_type, cpuParams) = obj.param;
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
        result << "cropsBegin=" << ov::test::utils::vec2str(cropsBegin) << "_";
        result << "cropsEnd=" << ov::test::utils::vec2str(cropsEnd) << "_";
        result << "netPRC=" << model_type << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); i++) {
            const auto& parameter = parameters[i];
            ov::Tensor tensor;
            const auto& param_type = parameter->get_output_element_type(0);
            const auto& static_shape = targetInputStaticShapes[i];
            switch (i) {
            case 0: {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 2560;
                    in_data.resolution = 256;
                    tensor = ov::test::utils::create_and_fill_tensor(param_type, static_shape, in_data);
                break;
            }
            case 1: {
                ASSERT_EQ(ov::shape_size(static_shape), blockShape.size());
                tensor = ov::Tensor(param_type, static_shape, blockShape.data());
                break;
            }
            case 2: {
                ASSERT_EQ(ov::shape_size(static_shape), cropsBegin.size());
                tensor = ov::Tensor(param_type, static_shape, cropsBegin.data());
                break;
            }
            case 3: {
                ASSERT_EQ(ov::shape_size(static_shape), cropsEnd.size());
                tensor = ov::Tensor(param_type, static_shape, cropsEnd.data());
                break;
            }
            default: {
                throw std::runtime_error("Incorrect parameter number!");
            }
            }
            inputs.insert({parameter, tensor});
        }
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        ov::element::Type model_type;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, blockShape, cropsBegin, cropsEnd, model_type, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        init_input_shapes(inputShapes);

        if (model_type == ov::element::Type_t::u8) {
            selectedType = std::string("ref_any_") + "I8";
        } else {
            std::string type_name = model_type.get_type_name();
            if (type_name == "f16")
                type_name = "fp16";
            if (type_name == "f32")
                type_name = "fp32";
            if (type_name == "f64")
                type_name = "fp64";
            std::transform(type_name.begin(), type_name.end(), type_name.begin(), ::toupper);
            selectedType = std::string("ref_any_") + type_name;
        }

        std::shared_ptr<ov::op::v0::Parameter> in0, in1, in2, in3;
        in0 = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
        in1 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::i64, inputDynamicShapes[1]);
        in2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::i64, inputDynamicShapes[2]);
        in3 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::i64, inputDynamicShapes[3]);
        auto btsNode = std::make_shared<ov::op::v1::BatchToSpace>(in0, in1, in2, in3);
        btsNode->get_rt_info() = getCPUInfo();
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(btsNode)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{in0, in1, in2, in3}, "BatchToSpace");
    }
};

TEST_P(BatchToSpaceCPULayerTest, CompareWithRefs) {
    run();
    // CheckPluginRelatedResults(compiledModel, "BatchToSpace");
};

namespace {

const std::vector<ov::element::Type> model_types = {ov::element::Type_t::u8,
                                                    ov::element::Type_t::i8,
                                                    ov::element::Type_t::i32,
                                                    ov::element::Type_t::f32,
                                                    ov::element::Type_t::bf16};

const std::vector<std::vector<int64_t>> blockShape4D1 = {{1, 1, 1, 2}, {1, 2, 2, 1}};
const std::vector<std::vector<int64_t>> cropsBegin4D1 = {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 2, 0}};
const std::vector<std::vector<int64_t>> cropsEnd4D1 = {{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 1, 1}};

std::vector<std::vector<ov::Shape>> staticInputShapes4D1 = {{{8, 16, 10, 10}, {4}, {4}, {4}}};

std::vector<std::vector<InputShape>> dynamicInputShapes4D1 = {
    {{{-1, -1, -1, -1}, {{8, 8, 6, 7}, {4, 10, 5, 5}, {12, 9, 7, 5}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}},
    {{{{4, 12}, {8, 16}, 6, -1}, {{8, 8, 6, 7}, {4, 10, 6, 5}, {12, 9, 6, 5}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes4D1Blocked = {
    {{{-1, 16, -1, -1}, {{4, 16, 5, 8}, {8, 16, 7, 6}, {12, 16, 4, 5}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

const std::vector<std::vector<int64_t>> blockShape4D2 = {{1, 2, 3, 4}, {1, 3, 4, 2}};
const std::vector<std::vector<int64_t>> cropsBegin4D2 = {{0, 0, 0, 1}, {0, 0, 1, 2}};
const std::vector<std::vector<int64_t>> cropsEnd4D2 = {{0, 0, 1, 0}, {0, 0, 3, 1}};

std::vector<std::vector<ov::Shape>> staticInputShapes4D2 = {{{24, 16, 7, 8}, {4}, {4}, {4}}};

std::vector<std::vector<InputShape>> dynamicInputShapes4D2 = {
    {{{-1, -1, -1, -1}, {{48, 4, 7, 8}, {24, 8, 6, 7}, {24, 16, 5, 5}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}},
    {{{24, {4, 10}, -1, -1}, {{24, 8, 6, 7}, {24, 6, 7, 5}, {24, 4, 5, 5}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes4D2Blocked = {
    {{{-1, 16, -1, -1}, {{24, 16, 5, 5}, {24, 16, 6, 7}, {48, 16, 4, 4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}},
     {{4}, {{4}, {4}, {4}}}}};

const std::vector<CPUSpecificParams> cpuParamsWithBlock_4D = {CPUSpecificParams({nChw16c}, {nChw16c}, {}, {}),
                                                              CPUSpecificParams({nChw8c}, {nChw8c}, {}, {}),
                                                              CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
                                                              CPUSpecificParams({nchw}, {nchw}, {}, {})};

const std::vector<CPUSpecificParams> cpuParams_4D = {CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
                                                     CPUSpecificParams({nchw}, {nchw}, {}, {})};

const auto staticBatchToSpaceParamsSet4D1 =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D1)),
                       ::testing::ValuesIn(blockShape4D1),
                       ::testing::ValuesIn(cropsBegin4D1),
                       ::testing::ValuesIn(cropsEnd4D1),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto dynamicBatchToSpaceParamsSet4D1 = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D1),
                                                                ::testing::ValuesIn(blockShape4D1),
                                                                ::testing::ValuesIn(cropsBegin4D1),
                                                                ::testing::ValuesIn(cropsEnd4D1),
                                                                ::testing::ValuesIn(model_types),
                                                                ::testing::ValuesIn(cpuParams_4D));

const auto dynamicBatchToSpaceParamsWithBlockedSet4D1 =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D1Blocked),
                       ::testing::ValuesIn(blockShape4D1),
                       ::testing::ValuesIn(cropsBegin4D1),
                       ::testing::ValuesIn(cropsEnd4D1),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto staticBatchToSpaceParamsSet4D2 =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes4D2)),
                       ::testing::ValuesIn(blockShape4D2),
                       ::testing::ValuesIn(cropsBegin4D2),
                       ::testing::ValuesIn(cropsEnd4D2),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_4D));

const auto dynamicBatchToSpaceParamsSet4D2 = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D2),
                                                                ::testing::ValuesIn(blockShape4D2),
                                                                ::testing::ValuesIn(cropsBegin4D2),
                                                                ::testing::ValuesIn(cropsEnd4D2),
                                                                ::testing::ValuesIn(model_types),
                                                                ::testing::ValuesIn(cpuParams_4D));

const auto dynamicBatchToSpaceParamsWithBlockedSet4D2 =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes4D2Blocked),
                       ::testing::ValuesIn(blockShape4D2),
                       ::testing::ValuesIn(cropsBegin4D2),
                       ::testing::ValuesIn(cropsEnd4D2),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_4D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase1_4D,
                         BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet4D1,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase1_4D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet4D1,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked1_4D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet4D1,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase2_4D,
                         BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet4D2,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase2_4D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet4D2,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked2_4D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet4D2,
                         BatchToSpaceCPULayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> blockShape5D1 = {{1, 1, 2, 2, 1}, {1, 2, 1, 2, 2}};
const std::vector<std::vector<int64_t>> cropsBegin5D1 = {{0, 0, 0, 0, 0}, {0, 0, 0, 3, 3}};
const std::vector<std::vector<int64_t>> cropsEnd5D1 = {{0, 0, 0, 0, 0}, {0, 0, 1, 0, 1}};

std::vector<std::vector<ov::Shape>> staticInputShapes5D1 = {{{8, 16, 4, 10, 10}, {5}, {5}, {5}}};

std::vector<std::vector<InputShape>> dynamicInputShapes5D1 = {
    {{{-1, -1, -1, -1, -1}, {{8, 16, 4, 10, 10}, {16, 10, 5, 11, 9}, {24, 6, 6, 8, 8}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}},
    {{{{8, 16}, {8, 16}, {2, 7}, -1, -1}, {{8, 16, 2, 6, 8}, {8, 10, 4, 7, 5}, {16, 8, 7, 5, 10}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes5D1Blocked = {
    {{{-1, 16, -1, -1, -1}, {{24, 16, 3, 6, 7}, {48, 16, 4, 5, 5}, {24, 16, 5, 8, 5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}}};

const std::vector<std::vector<int64_t>> blockShape5D2 = {{1, 2, 4, 3, 1}, {1, 1, 2, 4, 3}};
const std::vector<std::vector<int64_t>> cropsBegin5D2 = {{0, 0, 1, 2, 0}, {0, 0, 1, 0, 1}};
const std::vector<std::vector<int64_t>> cropsEnd5D2 = {{0, 0, 1, 0, 1}, {0, 0, 1, 1, 1}};

std::vector<std::vector<ov::Shape>> staticInputShapes5D2 = {{{48, 16, 3, 3, 3}, {5}, {5}, {5}}};

std::vector<std::vector<InputShape>> dynamicInputShapes5D2 = {
    {{{-1, -1, -1, -1, -1}, {{48, 4, 3, 3, 3}, {24, 16, 5, 3, 5}, {24, 8, 7, 5, 5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}},
    {{{24, {8, 16}, {3, 5}, -1, -1}, {{24, 16, 3, 4, 3}, {24, 12, 5, 3, 5}, {24, 8, 4, 5, 5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}},
    {// special case
     {{{1, 24}, {1, 16}, {1, 10}, {1, 10}, {1, 10}}, {{24, 16, 5, 3, 5}, {24, 16, 5, 3, 5}, {24, 16, 7, 5, 5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}}};

std::vector<std::vector<InputShape>> dynamicInputShapes5D2Blocked = {
    {{{-1, 16, -1, -1, -1}, {{24, 16, 4, 5, 5}, {48, 16, 3, 4, 3}, {24, 16, 5, 3, 5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}},
     {{5}, {{5}, {5}, {5}}}}};

const std::vector<CPUSpecificParams> cpuParamsWithBlock_5D = {CPUSpecificParams({nCdhw16c}, {nCdhw16c}, {}, {}),
                                                              CPUSpecificParams({nCdhw8c}, {nCdhw8c}, {}, {}),
                                                              CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
                                                              CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})};

const std::vector<CPUSpecificParams> cpuParams_5D = {CPUSpecificParams({ndhwc}, {ndhwc}, {}, {}),
                                                     CPUSpecificParams({ncdhw}, {ncdhw}, {}, {})};

const auto staticBatchToSpaceParamsSet5D1 =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes5D1)),
                       ::testing::ValuesIn(blockShape5D1),
                       ::testing::ValuesIn(cropsBegin5D1),
                       ::testing::ValuesIn(cropsEnd5D1),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto dynamicBatchToSpaceParamsSet5D1 = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D1),
                                                                ::testing::ValuesIn(blockShape5D1),
                                                                ::testing::ValuesIn(cropsBegin5D1),
                                                                ::testing::ValuesIn(cropsEnd5D1),
                                                                ::testing::ValuesIn(model_types),
                                                                ::testing::ValuesIn(cpuParams_5D));

const auto dynamicBatchToSpaceParamsWithBlockedSet5D1 =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D1Blocked),
                       ::testing::ValuesIn(blockShape5D1),
                       ::testing::ValuesIn(cropsBegin5D1),
                       ::testing::ValuesIn(cropsEnd5D1),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto staticBatchToSpaceParamsSet5D2 =
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapes5D2)),
                       ::testing::ValuesIn(blockShape5D2),
                       ::testing::ValuesIn(cropsBegin5D2),
                       ::testing::ValuesIn(cropsEnd5D2),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_5D));

const auto dynamicBatchToSpaceParamsSet5D2 = ::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D2),
                                                                ::testing::ValuesIn(blockShape5D2),
                                                                ::testing::ValuesIn(cropsBegin5D2),
                                                                ::testing::ValuesIn(cropsEnd5D2),
                                                                ::testing::ValuesIn(model_types),
                                                                ::testing::ValuesIn(cpuParams_5D));

const auto dynamicBatchToSpaceParamsWithBlockedSet5D2 =
    ::testing::Combine(::testing::ValuesIn(dynamicInputShapes5D2Blocked),
                       ::testing::ValuesIn(blockShape5D2),
                       ::testing::ValuesIn(cropsBegin5D2),
                       ::testing::ValuesIn(cropsEnd5D2),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(cpuParamsWithBlock_5D));

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase1_5D,
                         BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet5D1,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase1_5D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet5D1,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked1_5D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet5D1,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_StaticBatchToSpaceCPULayerTestCase2_5D,
                         BatchToSpaceCPULayerTest,
                         staticBatchToSpaceParamsSet5D2,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCase2_5D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSet5D2,
                         BatchToSpaceCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseWithBlocked2_5D,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsWithBlockedSet5D2,
                         BatchToSpaceCPULayerTest::getTestCaseName);

std::vector<InputShape> dynamicInputShapesZeroDimOutput = {
    {{-1, -1, -1, -1, -1}, {{2, 16, 1, 5, 5}}},
    {{5}, {{5}, {5}, {5}}},
    {{5}, {{5}, {5}, {5}}},
    {{5}, {{5}, {5}, {5}}}
};

const std::vector<int64_t> blockShapeZeroDimOutput = {1, 1, 2, 1, 1};
const std::vector<int64_t> cropsBeginZeroDimOutput = {0, 0, 0, 0, 0};
const std::vector<int64_t> cropsEndZeroDimOutput = {0, 0, 2, 0, 0};

const auto dynamicBatchToSpaceParamsSetZeroDimOutput = ::testing::Combine(::testing::Values(dynamicInputShapesZeroDimOutput),
                                                                          ::testing::Values(blockShapeZeroDimOutput),
                                                                          ::testing::Values(cropsBeginZeroDimOutput),
                                                                          ::testing::Values(cropsEndZeroDimOutput),
                                                                          ::testing::Values(ov::element::Type_t::f32),
                                                                          ::testing::ValuesIn(cpuParams_5D));

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseZeroDimOutput,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSetZeroDimOutput,
                         BatchToSpaceCPULayerTest::getTestCaseName);

std::vector<InputShape> dynamicInputShapesOutputDimOne = {
    {{-1, -1, -1, -1, -1}, {{2, 16, 1, 5, 5}}},
    {{5}, {{5}, {5}, {5}}},
    {{5}, {{5}, {5}, {5}}},
    {{5}, {{5}, {5}, {5}}}
};

const std::vector<int64_t> blockShapeOutputDimOne = {1, 1, 2, 1, 1};
const std::vector<int64_t> cropsBeginOutputDimOne = {0, 0, 0, 0, 0};
const std::vector<int64_t> cropsEndOutputDimOne = {0, 0, 1, 0, 0};

const auto dynamicBatchToSpaceParamsSetOutputDimOne = ::testing::Combine(::testing::Values(dynamicInputShapesOutputDimOne),
                                                                         ::testing::Values(blockShapeOutputDimOne),
                                                                         ::testing::Values(cropsBeginOutputDimOne),
                                                                         ::testing::Values(cropsEndOutputDimOne),
                                                                         ::testing::Values(ov::element::Type_t::f32),
                                                                         ::testing::ValuesIn(cpuParams_5D));

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBatchToSpaceCPULayerTestCaseOutputDimOne,
                         BatchToSpaceCPULayerTest,
                         dynamicBatchToSpaceParamsSetOutputDimOne,
                         BatchToSpaceCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
