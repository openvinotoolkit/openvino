// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

struct SliceScatterSpecificParams {
    std::vector<InputShape> updates;
    std::vector<int64_t> start;
    std::vector<int64_t> stop;
    std::vector<int64_t> step;
    std::vector<int64_t> axes;
};

typedef std::tuple<std::vector<InputShape>,          // Parameters shapes
                   SliceScatterSpecificParams,       // SliceScatter specific parameters
                   ov::test::utils::InputLayerType,  // Secondary input types
                   ElementType,                      // Network precision
                   CPUSpecificParams                 // CPU specific parameters
                   >
    SliceScatterLayerTestCPUParam;

class SliceScatterLayerCPUTest : public testing::WithParamInterface<SliceScatterLayerTestCPUParam>,
                                 virtual public SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SliceScatterLayerTestCPUParam> obj) {
        std::vector<InputShape> shapes;
        SliceScatterSpecificParams params;
        ov::test::utils::InputLayerType secondaryInputType;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(shapes, params, secondaryInputType, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : shapes) {
            for (const auto& item : shape.second) {
                result << ov::test::utils::vec2str(item) << "_";
            }
        }
        result << "start=" << ov::test::utils::vec2str(params.start) << "_";
        result << "stop=" << ov::test::utils::vec2str(params.stop) << "_";
        result << "step=" << ov::test::utils::vec2str(params.step) << "_";
        result << "axes=" << ov::test::utils::vec2str(params.axes) << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        std::vector<void*> inputValues = {sliceParams.start.data(),
                                          sliceParams.stop.data(),
                                          sliceParams.step.data(),
                                          sliceParams.axes.data()};

        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i <= 1u) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 1;
                in_data.range = 10;
                // Fill the slice input0 and input1 tensor with random data.
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                 targetInputStaticShapes[i],
                                                                 in_data);
            } else {
                // Fill the slice input2~input5 with specified data.
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i], inputValues[i - 2]};
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
    void SetUp() override {
        std::vector<InputShape> shapes;
        ov::test::utils::InputLayerType secondaryInputType;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(shapes, sliceParams, secondaryInputType, netPrecision, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::vector<InputShape> input_shapes = {shapes};
        input_shapes.push_back(sliceParams.updates[0]);
        init_input_shapes({input_shapes});
        for (auto& targetShapes : targetStaticShapes) {
            targetShapes.push_back({sliceParams.start.size()});
            targetShapes.push_back({sliceParams.stop.size()});
            targetShapes.push_back({sliceParams.step.size()});
            if (!sliceParams.axes.empty())
                targetShapes.push_back({sliceParams.axes.size()});
        }

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
        }
        std::shared_ptr<ov::Node> sliceNode;
        if (secondaryInputType == ov::test::utils::InputLayerType::PARAMETER) {
            // Slice start, stop, step, axes are parameters.
            auto startNode =
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{sliceParams.start.size()});
            auto stopdNode =
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{sliceParams.stop.size()});
            auto stepNode =
                std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{sliceParams.step.size()});

            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(startNode));
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(stopdNode));
            params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(stepNode));
            if (!sliceParams.axes.empty()) {
                // With axes parameter
                auto axesNode =
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{sliceParams.axes.size()});
                params.push_back(ov::as_type_ptr<ov::op::v0::Parameter>(axesNode));
                sliceNode = std::make_shared<ov::op::v15::SliceScatter>(params[0],
                                                                        params[1],
                                                                        startNode,
                                                                        stopdNode,
                                                                        stepNode,
                                                                        axesNode);
            } else {
                // without axes parameter
                sliceNode =
                    std::make_shared<ov::op::v15::SliceScatter>(params[0], params[1], startNode, stopdNode, stepNode);
            }
        } else if (secondaryInputType == ov::test::utils::InputLayerType::CONSTANT) {
            // Slice start, stop, step, axes are const.
            ov::Shape constShape = {sliceParams.start.size()};
            auto beginNode =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, sliceParams.start.data());
            auto endNode =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, sliceParams.stop.data());
            auto strideNode =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, sliceParams.step.data());
            if (!sliceParams.axes.empty()) {
                // With axes parameter
                auto axesNode =
                    std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, sliceParams.axes.data());
                sliceNode = std::make_shared<ov::op::v15::SliceScatter>(params[0],
                                                                        params[1],
                                                                        beginNode,
                                                                        endNode,
                                                                        strideNode,
                                                                        axesNode);
            } else {
                // without axes parameter
                sliceNode =
                    std::make_shared<ov::op::v15::SliceScatter>(params[0], params[1], beginNode, endNode, strideNode);
            }
        } else {
            // Not supported others.
            OPENVINO_THROW("SliceScatterLayerCPUTest: Unsupported ov::test::utils::InputLayerType , value: ",
                           secondaryInputType);
        }

        function = makeNgraphFunction(netPrecision, params, sliceNode, "SliceScatter");
    }
    SliceScatterSpecificParams sliceParams;
};

TEST_P(SliceScatterLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "SliceScatter");
}

namespace {

const std::vector<ov::test::utils::InputLayerType> inputLayerTypes = {ov::test::utils::InputLayerType::CONSTANT,
                                                                      ov::test::utils::InputLayerType::PARAMETER};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, {}};

const std::vector<ElementType> inputPrecisions = {ElementType::f32, ElementType::bf16, ElementType::i8};

const std::vector<std::vector<InputShape>> inputShapesDynamic2D = {
    {{// Origin dynamic shape
      {-1, -1},
      {// Dynamic shapes instances
       {32, 16},
       {16, 16},
       {24, 16}}}},
    {{// Origin dynamic shape
      {-1, 16},
      {// Dynamic shapes instances
       {16, 16},
       {20, 16},
       {32, 16}}}},
    {{// Origin dynamic shape
      {{16, 32}, {16, 32}},
      {// Dynamic shapes instances
       {16, 32},
       {32, 16},
       {24, 24}}}},
};

const std::vector<SliceScatterSpecificParams> paramsPlain2D = {
    SliceScatterSpecificParams{{{{-1, -1}, {{16, 6}}}}, {0, 10}, {16, 16}, {1, 1}, {0, 1}},
    SliceScatterSpecificParams{{{{-1, 3}, {{14, 3}}}}, {2, 5}, {16, 8}, {1, 1}, {}},
    SliceScatterSpecificParams{{{{{{14, 32}, {3, 8}}}, {{14, 6}}}}, {2, 5}, {16, 16}, {1, 2}, {0, 1}},
    SliceScatterSpecificParams{{{{-1, -1}, {{8, 16}}}}, {0, 0}, {16, 16}, {1, 2}, {1, 0}},
    SliceScatterSpecificParams{{{{8, -1}, {{8, 8}}}}, {0, 8}, {16, 16}, {2, 1}, {1, 0}},
};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Static_2D,
                         SliceScatterLayerCPUTest,
                         ::testing::Combine(::testing::Values(static_shapes_to_test_representation({{32, 16}})),
                                            ::testing::ValuesIn(paramsPlain2D),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(emptyCPUSpec)),
                         SliceScatterLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Dynamic_2D,
                         SliceScatterLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic2D),
                                            ::testing::ValuesIn(paramsPlain2D),
                                            ::testing::ValuesIn(inputLayerTypes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(emptyCPUSpec)),
                         SliceScatterLayerCPUTest::getTestCaseName);

const std::vector<SliceScatterSpecificParams> testCasesCommon4D = {
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{1, 2, 23, 23}}}}, {0, 2, 5, 4}, {1, 4, 28, 27}, {1, 1, 1, 1}, {}},
    SliceScatterSpecificParams{{{{1, 2, 32, 20}, {{1, 2, 32, 20}}}},
                               {0, 1, 0, 0},
                               {20, 3, 32, 1},
                               {1, 1, 1, 1},
                               {3, 1, 2, 0}},
    SliceScatterSpecificParams{{{{-1, {1, 4}, -1, {-1, 20}}, {{1, 3, 10, 20}}}},
                               {0, 0, 10, 0},
                               {1, 3, 20, 20},
                               {1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{-1, 5, {4, 6}, {-1, 3}}, {{1, 5, 5, 3}}}},
                               {0, 0, 20, 20},
                               {1, 5, 26, 25},
                               {1, 1, 2, 1},
                               {0, 1, 3, 2}},
    SliceScatterSpecificParams{{{{{1, 3}, {1, 2}, {15, -1}, {9, 11}}, {{1, 2, 15, 10}}}},
                               {0, 0, 0, 20},
                               {1, 2, 30, 30},
                               {1, 1, 2, 1},
                               {}},
    SliceScatterSpecificParams{{{{-1, 3, 30, 10}, {{1, 3, 30, 10}}}},
                               {0, 0, 2, 10},
                               {1, 3, 32, 20},
                               {1, 1, 1, 1},
                               {0, 1, 2, 3}},
    SliceScatterSpecificParams{{{{{0, 64}, 4, 32, 20}, {{1, 4, 32, 20}}}},
                               {0, 1, 0, 10},
                               {1, 5, 32, 30},
                               {1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{1, 4, 30, 4}}}},
                               {0, 1, 2, 10},
                               {1, 5, 32, 18},
                               {1, 1, 1, 2},
                               {0, 1, 2, 3}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{1, 3, 30, 4}}}}, {0, 0, 2, 10}, {1, 8, 32, 18}, {1, 2, 1, 2}, {}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{1, 5, 32, 8}}}}, {0, 0, 10}, {25, 32, 18}, {5, 1, 1}, {0, 2, 3}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{1, 5, 8, 32}}}}, {15, 0, 0}, {30, 32, 32}, {2, 15, 1}, {2, 0, 3}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1}, {{0, 5, 8, 0}}}},
                               {15, 32, 64},
                               {30, 32, 128},
                               {2, 15, 1},
                               {2, 0, 3}}};

const std::vector<std::vector<ov::Shape>> inputShapesStatic4D = {{{1, 5, 32, 32}}, {{2, 5, 32, 48}}};

const std::vector<std::vector<InputShape>> inputShapesDynamic4D = {{{// Origin dynamic shape
                                                                     {-1, -1, -1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {1, 5, 32, 32},
                                                                      {2, 5, 32, 32},
                                                                      {1, 5, 64, 32}}}},
                                                                   {{// Origin dynamic shape
                                                                     {-1, 5, -1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {1, 5, 32, 32},
                                                                      {2, 5, 32, 32},
                                                                      {3, 5, 32, 32}}}},
                                                                   {{// Origin dynamic shape
                                                                     {{1, 5}, 5, {32, 64}, {32, 64}},
                                                                     {// Dynamic shapes instances
                                                                      {2, 5, 32, 32},
                                                                      {1, 5, 48, 32},
                                                                      {5, 5, 32, 32}}}}};

const std::vector<CPUSpecificParams> CPUParamsCommon4D = {
    cpuParams_nchw,
    cpuParams_nhwc,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs_Common_Static_4D,
    SliceScatterLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic4D)),
                       ::testing::ValuesIn(testCasesCommon4D),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsCommon4D)),
    SliceScatterLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D,
                         SliceScatterLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic4D),
                                            ::testing::ValuesIn(testCasesCommon4D),
                                            ::testing::ValuesIn(inputLayerTypes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsCommon4D)),
                         SliceScatterLayerCPUTest::getTestCaseName);

const std::vector<SliceScatterSpecificParams> testCasesCommon5D = {
    SliceScatterSpecificParams{{{{-1, -1, -1, -1, -1}, {{1, 2, 5, 23, 23}}}},
                               {0, 2, 0, 5, 4},
                               {1, 4, 5, 28, 27},
                               {1, 1, 1, 1, 1},
                               {0, 1, 2, 3, 4}},
    SliceScatterSpecificParams{{{{1, 5, 10, 32, 20}, {{1, 5, 10, 32, 20}}}},
                               {0, 0, 10, 0, 0},
                               {1, 5, 20, 32, 20},
                               {1, 1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{{0, 1}, {1, 3}, {5, 15}, {30, -1}, {-1, 20}}, {{1, 2, 10, 32, 20}}}},
                               {0, 1, 10, 0, 0},
                               {20, 3, 20, 32, 1},
                               {1, 1, 1, 1, 1},
                               {4, 1, 2, 3, 0}},
    SliceScatterSpecificParams{{{{-1, -1, 20, 10, 3}, {{1, 3, 20, 10, 3}}}},
                               {0, 20, 0, 0, 20},
                               {1, 30, 20, 5, 26},
                               {1, 1, 1, 2, 2},
                               {0, 3, 2, 1, 4}},
    SliceScatterSpecificParams{{{{{0, 128}, 2, 5, 30, 10}, {{1, 2, 5, 30, 10}}}},
                               {0, 0, 10, 0, 20},
                               {1, 2, 20, 30, 30},
                               {1, 1, 2, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1, -1}, {{1, 5, 8, 22, 20}}}},
                               {0, 0, 2, 10, 0},
                               {1, 5, 10, 32, 20},
                               {1, 1, 1, 1, 1},
                               {0, 1, 2, 3, 4}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1, -1}, {{1, 4, 20, 22, 32}}}},
                               {0, 1, 0, 10, 0},
                               {1, 5, 20, 32, 32},
                               {1, 1, 1, 1, 1},
                               {}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1, -1}, {{1, 5, 5, 16, 16}}}},
                               {0, 0, 0, 0, 0},
                               {1, 5, 10, 16, 16},
                               {1, 1, 2, 1, 1},
                               {0, 1, 2, 3, 4}}};

const std::vector<std::vector<ov::Shape>> inputShapesStatic5D = {{{1, 5, 20, 32, 32}}, {{2, 5, 32, 32, 32}}};

const std::vector<std::vector<InputShape>> inputShapesDynamic5D = {{{// Origin dynamic shape
                                                                     {-1, -1, -1, -1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {1, 5, 32, 32, 32},
                                                                      {1, 5, 32, 32, 48},
                                                                      {1, 5, 64, 64, 64},
                                                                      {1, 10, 32, 32, 32}}}},
                                                                   {{// Origin dynamic shape
                                                                     {-1, 5, -1, -1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {1, 5, 32, 32, 48},
                                                                      {1, 5, 32, 48, 32},
                                                                      {1, 5, 48, 32, 32}}}},
                                                                   {{// Origin dynamic shape
                                                                     {{1, 5}, 5, {32, 64}, {32, 64}, {32, 64}},
                                                                     {// Dynamic shapes instances
                                                                      {2, 5, 32, 32, 32},
                                                                      {1, 5, 48, 32, 32},
                                                                      {5, 5, 32, 32, 48}}}}};

const std::vector<CPUSpecificParams> CPUParamsCommon5D = {
    cpuParams_ncdhw,
    cpuParams_ndhwc,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs_Common_Static_5D,
    SliceScatterLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic5D)),
                       ::testing::ValuesIn(testCasesCommon5D),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsCommon5D)),
    SliceScatterLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D,
                         SliceScatterLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic5D),
                                            ::testing::ValuesIn(testCasesCommon5D),
                                            ::testing::ValuesIn(inputLayerTypes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsCommon5D)),
                         SliceScatterLayerCPUTest::getTestCaseName);

const std::vector<SliceScatterSpecificParams> testCasesFullSlice5D = {
    SliceScatterSpecificParams{{{{-1, -1, -1, -1, -1}, {{1, 5, 32, 32, 32}}}}, {}, {}, {}, {}},
    SliceScatterSpecificParams{{{{-1, -1, -1, -1, -1}, {{1, 5, 32, 32, 32}}}}, {-64}, {64}, {1}, {-1}},
    SliceScatterSpecificParams{{{{1, 5, 32, 32, 32}, {{1, 5, 32, 32, 32}}}},
                               {-64, 0, -32, 32},
                               {64, 33, 32, -33},
                               {1, 1, 1, -1},
                               {4, -2, 2, 1}},
};

const std::vector<std::vector<InputShape>> inputShapesFullSlice5D = {
    {{{1, 5, 32, 32, 32}, {{1, 5, 32, 32, 32}}}},
    {{{-1, -1, -1, -1, -1}, {{1, 5, 32, 32, 32}}}},
    {{{-1, 5, -1, -1, -1}, {{1, 5, 32, 32, 32}}}},
    {{{-1, 5, 32, {31, 64}, 32}, {{1, 5, 32, 32, 32}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Full_Slice_5D,
                         SliceScatterLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesFullSlice5D),
                                            ::testing::ValuesIn(testCasesFullSlice5D),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsCommon5D)),
                         SliceScatterLayerCPUTest::getTestCaseName);
}  // namespace
