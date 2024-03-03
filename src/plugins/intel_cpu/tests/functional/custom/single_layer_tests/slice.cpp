// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/test_enums.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

struct Slice8SpecificParams {
    std::vector<int64_t> start;
    std::vector<int64_t> stop;
    std::vector<int64_t> step;
    std::vector<int64_t> axes;
};

typedef std::tuple<std::vector<InputShape>,          // Parameters shapes
                   Slice8SpecificParams,             // Slice-8 specific parameters
                   ov::test::utils::InputLayerType,  // Secondary input types
                   ElementType,                      // Network precision
                   CPUSpecificParams                 // CPU specific parameters
                   >
    Slice8LayerTestCPUParam;

class Slice8LayerCPUTest : public testing::WithParamInterface<Slice8LayerTestCPUParam>,
                           virtual public SubgraphBaseTest,
                           public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<Slice8LayerTestCPUParam> obj) {
        std::vector<InputShape> shapes;
        Slice8SpecificParams params;
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
            if (i == 0u) {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 1;
                in_data.range = 10;
                // Fill the slice input0 tensor with random data.
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            } else {
                // Fill the slice input1~input4 with specified data.
                tensor = ov::Tensor{ov::element::i64, targetInputStaticShapes[i], inputValues[i - 1]};
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

            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(startNode));
            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(stopdNode));
            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(stepNode));
            if (!sliceParams.axes.empty()) {
                // With axes parameter
                auto axesNode =
                    std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{sliceParams.axes.size()});
                params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(axesNode));
                sliceNode = std::make_shared<ov::op::v8::Slice>(params[0], startNode, stopdNode, stepNode, axesNode);
            } else {
                // without axes parameter
                sliceNode = std::make_shared<ov::op::v8::Slice>(params[0], startNode, stopdNode, stepNode);
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
                sliceNode = std::make_shared<ov::op::v8::Slice>(params[0], beginNode, endNode, strideNode, axesNode);
            } else {
                // without axes parameter
                sliceNode = std::make_shared<ov::op::v8::Slice>(params[0], beginNode, endNode, strideNode);
            }
        } else {
            // Not supported others.
            OPENVINO_THROW("Slice8LayerCPUTest: Unsupported ov::test::utils::InputLayerType , value: ",
                           secondaryInputType);
        }

        function = makeNgraphFunction(netPrecision, params, sliceNode, "Slice8");
    }
    Slice8SpecificParams sliceParams;
};

TEST_P(Slice8LayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Slice8");
}

namespace {

const std::vector<ov::test::utils::InputLayerType> inputLayerTypes = {ov::test::utils::InputLayerType::CONSTANT,
                                                                      ov::test::utils::InputLayerType::PARAMETER};

const auto cpuParams_nChw16c = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, {}};
const auto cpuParams_nCdhw16c = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, {}};

const auto cpuParams_nChw8c = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, {}};
const auto cpuParams_nCdhw8c = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, {}};

const auto cpuParams_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {}, {}};
const auto cpuParams_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, {}};

const auto cpuParams_nchw = CPUSpecificParams{{nchw}, {nchw}, {}, {}};
const auto cpuParams_ncdhw = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, {}};

const std::vector<ElementType> inputPrecisions = {ElementType::f32, ElementType::bf16, ElementType::i8};

const std::vector<std::vector<InputShape>> inputShapesDynamic2D = {{{// Origin dynamic shape
                                                                     {-1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {32, 20},
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
                                                                      {24, 24}}}}};

const std::vector<Slice8SpecificParams> paramsPlain2D = {Slice8SpecificParams{{0, 10}, {16, 16}, {1, 1}, {0, 1}},
                                                         Slice8SpecificParams{{2, 5}, {16, 8}, {1, 1}, {}},
                                                         Slice8SpecificParams{{2, 5}, {16, 16}, {1, 2}, {0, 1}},
                                                         Slice8SpecificParams{{0, 0}, {16, 16}, {1, 2}, {1, 0}},
                                                         Slice8SpecificParams{{0}, {16}, {2}, {0}},
                                                         Slice8SpecificParams{{0}, {16}, {1}, {1}}};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Static_2D,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::Values(static_shapes_to_test_representation({{32, 20}})),
                                            ::testing::ValuesIn(paramsPlain2D),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(emptyCPUSpec)),
                         Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Plain_Dynamic_2D,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic2D),
                                            ::testing::ValuesIn(paramsPlain2D),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(emptyCPUSpec)),
                         Slice8LayerCPUTest::getTestCaseName);

const std::vector<Slice8SpecificParams> testCasesCommon4D = {
    Slice8SpecificParams{{0, 2, 5, 4}, {1, 4, 28, 27}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 1, 0, 0}, {20, 3, 32, 1}, {1, 1, 1, 1}, {3, 1, 2, 0}},
    Slice8SpecificParams{{0, 0, 10, 0}, {1, 3, 20, 20}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 20, 20}, {1, 5, 26, 25}, {1, 1, 2, 1}, {0, 1, 3, 2}},
    Slice8SpecificParams{{0, 0, 0, 20}, {1, 2, 30, 30}, {1, 1, 2, 1}, {}},
    Slice8SpecificParams{{0, 0, 2, 10}, {1, 3, 32, 20}, {1, 1, 1, 1}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 1, 0, 10}, {1, 5, 32, 30}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 1, 2, 10}, {1, 5, 32, 18}, {1, 1, 1, 2}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 0, 2, 10}, {1, 8, 32, 18}, {1, 2, 1, 2}, {}},
    Slice8SpecificParams{{0, 0, 10}, {2, 32, 18}, {1, 1, 1}, {1, 2, 3}},
    Slice8SpecificParams{{0, 10}, {2, 32}, {1, 1}, {1, 3}}};

const std::vector<std::vector<ov::Shape>> inputShapesStatic4D = {{{1, 5, 32, 32}}, {{2, 5, 32, 48}}};

const std::vector<std::vector<InputShape>> inputShapesDynamic4D = {{{// Origin dynamic shape
                                                                     {-1, -1, -1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {1, 5, 32, 32},
                                                                      {2, 5, 32, 32},
                                                                      {1, 5, 64, 64}}}},
                                                                   {{// Origin dynamic shape
                                                                     {-1, 5, -1, -1},
                                                                     {// Dynamic shapes instances
                                                                      {1, 5, 32, 32},
                                                                      {2, 5, 32, 32},
                                                                      {3, 5, 32, 36}}}},
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
    Slice8LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic4D)),
                       ::testing::ValuesIn(testCasesCommon4D),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsCommon4D)),
    Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic4D),
                                            ::testing::ValuesIn(testCasesCommon4D),
                                            ::testing::ValuesIn(inputLayerTypes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsCommon4D)),
                         Slice8LayerCPUTest::getTestCaseName);

const std::vector<Slice8SpecificParams> testCasesBlocked4DSubset1 = {
    Slice8SpecificParams{{0, 0, 0, 0}, {1, 32, 32, 32}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 16, 0}, {1, 32, 32, 32}, {1, 1, 1, 1}, {0, 3, 2, 1}},
    Slice8SpecificParams{{0, 0, 0}, {32, 32, 16}, {1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 0}, {16, 32, 32}, {1, 1, 1}, {1, 3, 2}},
};

const std::vector<Slice8SpecificParams> testCasesBlocked4DSubset2 = {
    Slice8SpecificParams{{0, 0, 5, 4}, {1, 16, 28, 27}, {1, 1, 1, 1}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 16, 0, 0}, {1, 32, 10, 10}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 10, 0}, {16, 1, 20, 10}, {1, 1, 1, 1}, {1, 0, 2, 3}},
    Slice8SpecificParams{{0, 0, 20, 20}, {1, 32, 25, 25}, {1, 1, 1, 1}, {0, 1, 3, 2}},
    Slice8SpecificParams{{0, 16, 0, 20}, {32, 32, 1, 30}, {1, 1, 1, 2}, {2, 1, 0, 3}},
    Slice8SpecificParams{{0, 16, 2, 10}, {1, 32, 32, 20}, {1, 1, 2, 1}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 16, 0, 0}, {2, 64, 32, 20}, {1, 1, 1, 1}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 32, 0, 0}, {2, 50, 32, 20}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 0, 0}, {32, 12, 2, 20}, {1, 1, 1, 1}, {0, 3, 2, 1}},
    Slice8SpecificParams{{0, -16, 0, 10}, {2, 100, 32, 20}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, -16, 0, 0}, {2, -4, 32, 20}, {1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, -32, 0, 0}, {2, -12, 32, 20}, {1, 1, 1, 1}, {}}};

const std::vector<std::vector<ov::Shape>> inputShapesBlockedStatic4DSubset1 = {{{1, 32, 32, 32}}, {{1, 32, 32, 64}}};

const std::vector<std::vector<ov::Shape>> inputShapesBlockedStatic4DSubset2 = {{{1, 64, 32, 32}}, {{1, 64, 32, 64}}};

const std::vector<std::vector<InputShape>> inputShapesBlockedDynamic4DSubset1 = {{{// Origin dynamic shape
                                                                                   {-1, 32, -1, -1},
                                                                                   {// Dynamic shapes instances
                                                                                    {1, 32, 32, 32},
                                                                                    {2, 32, 32, 32},
                                                                                    {3, 32, 32, 48}}}},
                                                                                 {{// Origin dynamic shape
                                                                                   {{1, 5}, 32, {32, 64}, {32, 64}},
                                                                                   {// Dynamic shapes instances
                                                                                    {2, 32, 32, 32},
                                                                                    {1, 32, 48, 32},
                                                                                    {5, 32, 32, 48}}}}};

const std::vector<std::vector<InputShape>> inputShapesBlockedDynamic4DSubset2 = {{{// Origin dynamic shape
                                                                                   {-1, 64, -1, -1},
                                                                                   {// Dynamic shapes instances
                                                                                    {1, 64, 64, 32},
                                                                                    {2, 64, 32, 32},
                                                                                    {3, 64, 32, 48}}}},
                                                                                 {{// Origin dynamic shape
                                                                                   {{1, 5}, 64, {32, 64}, {32, 64}},
                                                                                   {// Dynamic shapes instances
                                                                                    {2, 64, 32, 32},
                                                                                    {1, 64, 48, 32},
                                                                                    {1, 64, 64, 64}}}}};

const std::vector<CPUSpecificParams> CPUParamsBlocked4D = {
    cpuParams_nChw16c,
    cpuParams_nChw8c,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs_Common_Static_4D_Subset1,
    Slice8LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic4DSubset1)),
                       ::testing::ValuesIn(testCasesBlocked4DSubset1),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsBlocked4D)),
    Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D_Subset1,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBlockedDynamic4DSubset1),
                                            ::testing::ValuesIn(testCasesBlocked4DSubset1),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsBlocked4D)),
                         Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs_Common_Static_4D_Subset2,
    Slice8LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic4DSubset2)),
                       ::testing::ValuesIn(testCasesBlocked4DSubset2),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsBlocked4D)),
    Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_4D_Subset2,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBlockedDynamic4DSubset2),
                                            ::testing::ValuesIn(testCasesBlocked4DSubset2),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsBlocked4D)),
                         Slice8LayerCPUTest::getTestCaseName);

const std::vector<Slice8SpecificParams> testCasesCommon5D = {
    Slice8SpecificParams{{0, 2, 0, 5, 4}, {1, 4, 5, 28, 27}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 10, 0, 0}, {1, 5, 20, 32, 20}, {1, 1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 1, 10, 0, 0}, {20, 3, 20, 32, 1}, {1, 1, 1, 1, 1}, {4, 1, 2, 3, 0}},
    Slice8SpecificParams{{0, 20, 0, 0, 20}, {1, 30, 20, 5, 26}, {1, 1, 1, 2, 2}, {0, 3, 2, 1, 4}},
    Slice8SpecificParams{{0, 0, 10, 0, 20}, {1, 2, 20, 30, 30}, {1, 1, 2, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 2, 10, 0}, {1, 5, 10, 32, 20}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 1, 0, 10, 0}, {1, 5, 20, 32, 32}, {1, 1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 0, 0, 0}, {1, 5, 10, 16, 16}, {1, 1, 2, 1, 1}, {0, 1, 2, 3, 4}}};

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
    Slice8LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesStatic5D)),
                       ::testing::ValuesIn(testCasesCommon5D),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsCommon5D)),
    Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDynamic5D),
                                            ::testing::ValuesIn(testCasesCommon5D),
                                            ::testing::ValuesIn(inputLayerTypes),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsCommon5D)),
                         Slice8LayerCPUTest::getTestCaseName);

const std::vector<Slice8SpecificParams> testCasesBlocked5DSubset1 = {
    Slice8SpecificParams{{0, 0, 0, 5, 4}, {1, 16, 5, 28, 27}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 10, 0, 0}, {1, 16, 20, 32, 20}, {1, 1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 0, 20, 20}, {16, 1, 20, 26, 30}, {1, 1, 1, 2, 2}, {1, 0, 2, 4, 3}},
    Slice8SpecificParams{{0, 0, 10, 0, 20}, {1, 16, 20, 30, 30}, {1, 1, 2, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 2, 10, 0}, {1, 16, 10, 32, 20}, {1, 1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 0, 10, 0}, {1, 8, 20, 32, 32}, {1, 1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 0, 0, 0, 0}, {1, 16, 10, 16, 16}, {1, 1, 2, 1, 1}, {0, 1, 2, 3, 4}},
};

const std::vector<Slice8SpecificParams> testCasesBlocked5DSubset2 = {
    Slice8SpecificParams{{0, 0, 0, 5, 4}, {1, 16, 5, 28, 27}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 5, 4}, {16, 5, 28, 27}, {1, 1, 1, 1}, {1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 10, 0, 0}, {1, 16, 20, 32, 20}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 0, 20, 20}, {1, 20, 16, 30, 26}, {1, 1, 1, 2, 2}, {0, 2, 1, 3, 4}},
    Slice8SpecificParams{{0, 0, 10, 0, 20}, {1, 16, 20, 30, 30}, {1, 1, 2, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 2, 10, 0}, {1, 16, 10, 32, 20}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 0, 10, 0}, {1, 8, 20, 32, 32}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 0, 0, 0, 0}, {10, 16, 1, 16, 16}, {2, 1, 1, 1, 1}, {2, 1, 0, 3, 4}},
    Slice8SpecificParams{{0, 0, 0, 0, 0}, {1, 25, 20, 10, 10}, {1, 1, 1, 1, 1}, {}},
    Slice8SpecificParams{{0, 16, 0, 0, 0}, {1, 25, 20, 10, 10}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
    Slice8SpecificParams{{0, 16, 0, 0, 0}, {1, 64, 20, 10, 10}, {1, 1, 1, 1, 1}, {0, 1, 2, 3, 4}},
};

const std::vector<std::vector<ov::Shape>> inputShapesBlockedStatic5DSubset1 = {{{1, 16, 32, 32, 32}},
                                                                               {{2, 16, 32, 32, 32}},
                                                                               {{2, 32, 32, 32, 32}}};

const std::vector<std::vector<ov::Shape>> inputShapesBlockedStatic5DSubset2 = {{{1, 64, 32, 32, 32}},
                                                                               {{2, 64, 32, 64, 32}},
                                                                               {{2, 64, 32, 32, 32}}};

const std::vector<std::vector<InputShape>> inputShapesBlockedDynamic5DSubset1 = {
    {{// Origin dynamic shape
      {-1, 16, -1, -1, -1},
      {// Dynamic shapes instances
       {1, 16, 32, 32, 32},
       {2, 16, 32, 32, 32},
       {2, 16, 32, 32, 32}}}},
    {{// Origin dynamic shape
      {{1, 5}, 16, {16, 32}, {16, 32}, {16, 32}},
      {// Dynamic shapes instances
       {1, 16, 32, 32, 32},
       {2, 16, 32, 32, 32},
       {1, 16, 20, 32, 32}}}}};

const std::vector<std::vector<InputShape>> inputShapesBlockedDynamic5DSubset2 = {
    {
        {// Origin dynamic shape
         {-1, 64, -1, -1, -1},
         {// Dynamic shapes instances
          {1, 64, 64, 32, 32},
          {2, 64, 32, 32, 32},
          {3, 64, 32, 48, 32}}},
    },
    {{// Origin dynamic shape
      {{1, 5}, 64, {16, 32}, {16, 32}, {16, 32}},
      {// Dynamic shapes instances
       {1, 64, 32, 32, 32},
       {2, 64, 32, 32, 32},
       {1, 64, 20, 32, 32}}}}};

const std::vector<CPUSpecificParams> CPUParamsBlocked5D = {
    cpuParams_nCdhw16c,
    cpuParams_nCdhw8c,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs_Common_Static_5D_Subset1,
    Slice8LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic5DSubset1)),
                       ::testing::ValuesIn(testCasesBlocked5DSubset1),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsBlocked5D)),
    Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D_Subset1,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBlockedDynamic5DSubset1),
                                            ::testing::ValuesIn(testCasesBlocked5DSubset1),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsBlocked5D)),
                         Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs_Common_Static_5D_Subset2,
    Slice8LayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBlockedStatic4DSubset2)),
                       ::testing::ValuesIn(testCasesBlocked4DSubset2),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(CPUParamsBlocked4D)),
    Slice8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Common_Dynamic_5D_Subset2,
                         Slice8LayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesBlockedDynamic5DSubset2),
                                            ::testing::ValuesIn(testCasesBlocked5DSubset2),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(CPUParamsBlocked5D)),
                         Slice8LayerCPUTest::getTestCaseName);

/* Descriptors check */

class Slice8LayerDescriptorCPUTest : public Slice8LayerCPUTest {};

TEST_P(Slice8LayerDescriptorCPUTest, DescriptorsCheck) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ASSERT_THROW(compile_model(), ov::Exception);
}

const std::vector<Slice8SpecificParams> testCasesDescriptors = {
    Slice8SpecificParams{{0, -4, 0, 0}, {0, 2147483647, 0, 0}, {1, 1, 1, 1}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 5, 0, 0}, {1, 20, 28, 27}, {1, 1, 1, 1}, {0, 1, 2, 3}},
    Slice8SpecificParams{{0, 0, 0, 0}, {1, 2147483647, 32, 32}, {1, 2, 1, 1}, {0, 1, 2, 3}}};

const std::vector<std::vector<InputShape>> inputShapesDescriptors = {{{{},
                                                                       {// Static shapes
                                                                        {1, 16, 32, 32}}}},
                                                                     {{{},
                                                                       {// Static shapes
                                                                        {1, 17, 32, 32}}}},
                                                                     {{// Origin dynamic shapes
                                                                       {1, -1, 32, 32},
                                                                       {// Dynamic shapes instances
                                                                        {1, 16, 32, 32},
                                                                        {1, 32, 32, 32}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Slice8LayerDescriptorCPUTest,
                         Slice8LayerDescriptorCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesDescriptors),
                                            ::testing::ValuesIn(testCasesDescriptors),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(cpuParams_nChw8c)),
                         Slice8LayerDescriptorCPUTest::getTestCaseName);

}  // namespace