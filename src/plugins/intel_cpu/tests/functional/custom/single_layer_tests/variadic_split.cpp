// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/test_enums.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

using LengthsPerInfer = std::vector<std::vector<int>>;
typedef std::tuple<InputShape,
                   int64_t,                          // Axis
                   LengthsPerInfer,                  // Split lengths
                   ov::test::utils::InputLayerType,  // lengths input type
                   ElementType,                      // Net precision
                   CPUSpecificParams>
    varSplitCPUTestParams;

class VariadicSplitLayerCPUTest : public testing::WithParamInterface<varSplitCPUTestParams>,
                                  virtual public SubgraphBaseTest,
                                  public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<varSplitCPUTestParams> obj) {
        InputShape shapes;
        int64_t axis;
        LengthsPerInfer splitLengths;
        ov::test::utils::InputLayerType lengthsType;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(shapes, axis, splitLengths, lengthsType, netPrecision, cpuParams) = obj.param;

        std::ostringstream result;
        result << "IS=";
        result << ov::test::utils::partialShape2str({shapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : shapes.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << "axis=" << axis << "_";
        result << "splitLengths=(";
        for (const auto& lengths : splitLengths) {
            result << ov::test::utils::vec2str(lengths) << ",";
        }
        result << ")_lengthsType=" << lengthsType << "_";
        result << "netPRC=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputShape inputShapes;
        int64_t axis;
        ov::test::utils::InputLayerType lengthsType;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        std::tie(inputShapes, axis, lengthsPerInfer, lengthsType, netPrecision, cpuParams) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        selectedType += std::string("_") + ov::element::Type(netPrecision).to_string();

        std::vector<InputShape> shapesToInit{inputShapes};
        if (lengthsType == ov::test::utils::InputLayerType::PARAMETER) {
            std::vector<ov::Shape> lengthsStaticShapes(inputShapes.second.size(), {lengthsPerInfer[0].size()});
            shapesToInit.emplace_back(InputShape{{static_cast<int>(lengthsPerInfer[0].size())}, lengthsStaticShapes});
        }

        init_input_shapes(shapesToInit);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};

        std::shared_ptr<ov::Node> splitLengthsOp;
        if (lengthsType == ov::test::utils::InputLayerType::PARAMETER) {
            auto param =
                std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{lengthsPerInfer[0].size()});
            params.push_back(param);
            splitLengthsOp = param;
        } else {
            splitLengthsOp =
                ov::op::v0::Constant::create(ov::element::i32, {lengthsPerInfer[0].size()}, lengthsPerInfer[0]);
        }

        auto splitAxisOp = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
        auto varSplit = std::make_shared<ov::op::v1::VariadicSplit>(params[0], splitAxisOp, splitLengthsOp);
        varSplit->get_rt_info() = getCPUInfo();

        ov::ResultVector results;
        for (const auto& out : varSplit->outputs())
            results.push_back(std::make_shared<ov::op::v0::Result>(out));
        function = std::make_shared<ov::Model>(results, params, "VariadicSplitCPU");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const auto& dataPrecision = funcInputs[0].get_element_type();
        const auto& dataShape = targetInputStaticShapes.front();
        const auto dataTensor = ov::test::utils::create_and_fill_tensor(dataPrecision, dataShape);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), dataTensor});

        if (funcInputs.size() > 1) {
            const auto& curLengthsVals = lengthsPerInfer[inferRequestNum++ % lengthsPerInfer.size()];
            auto lengthsTensor = ov::Tensor(funcInputs[1].get_element_type(), targetInputStaticShapes[1]);
            OPENVINO_ASSERT(curLengthsVals.size() == lengthsTensor.get_size());

            auto* dataPtr = lengthsTensor.data<int>();
            for (size_t i = 0; i < lengthsTensor.get_size(); ++i) {
                dataPtr[i] = curLengthsVals[i];
            }
            inputs.insert({funcInputs[1].get_node_shared_ptr(), lengthsTensor});
        }
    }

private:
    size_t inferRequestNum = 0;
    LengthsPerInfer lengthsPerInfer;
};

TEST_P(VariadicSplitLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Split");
}

namespace {
const auto planar_4D_ref = CPUSpecificParams{{nchw}, {nchw}, {}, "ref"};
const auto planar_5D_ref = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "ref"};

const auto planar_4D = CPUSpecificParams{{nchw}, {nchw}, {}, "unknown"};
const auto planar_5D = CPUSpecificParams{{ncdhw}, {ncdhw}, {}, "unknown"};

const auto perChannels_4D = CPUSpecificParams{{nhwc}, {nhwc}, {}, "ref"};
const auto perChannels_5D = CPUSpecificParams{{ndhwc}, {ndhwc}, {}, "ref"};

const auto perChannelsToPlanar_4D = CPUSpecificParams{{nhwc}, {nchw}, {}, "ref"};
const auto perChannelsToPlanar_5D = CPUSpecificParams{{ndhwc}, {ncdhw}, {}, "ref"};

const auto blocked8_4D = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "unknown"};
const auto blocked8_5D = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "unknown"};

const auto blocked8_4D_ref = CPUSpecificParams{{nChw8c}, {nChw8c}, {}, "ref"};
const auto blocked8_5D_ref = CPUSpecificParams{{nCdhw8c}, {nCdhw8c}, {}, "ref"};

const auto blocked16_4D = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "unknown"};
const auto blocked16_5D = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "unknown"};

const auto blocked16_4D_ref = CPUSpecificParams{{nChw16c}, {nChw16c}, {}, "ref"};
const auto blocked16_5D_ref = CPUSpecificParams{{nCdhw16c}, {nCdhw16c}, {}, "ref"};

// List of precisions natively supported by onednn.
const std::vector<ElementType> netPrecisions = {ElementType::i8, ElementType::i32, ElementType::f32, ElementType::bf16};

const std::vector<ov::test::utils::InputLayerType> lengthsTypes = {ov::test::utils::InputLayerType::CONSTANT,
                                                                   ov::test::utils::InputLayerType::PARAMETER};

const std::vector<InputShape> inputShapes4D_Nspc2NcspSpecial = {
    {{}, {{3, 5, 24, 9}}},
    {// dynamic
     {-1, -1, -1, -1},
     // target
     {{1, 8, 5, 7}, {3, 9, 7, 9}, {5, 6, 1, 8}}},
    {// dynamic
     {{1, 5}, {1, 64}, {1, 25}, {2, 10}},
     // target
     {{2, 7, 5, 7}, {1, 10, 10, 2}, {3, 5, 6, 9}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Nspc2NcspSpecial,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_Nspc2NcspSpecial),
                                            ::testing::Values(1),
                                            ::testing::Values(LengthsPerInfer{{1, 2, -1, 1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(perChannelsToPlanar_4D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_Nspc2NcspSpecial = {
    {{}, {{3, 4, 7, 9, 3}}},
    {// dynamic
     {-1, -1, -1, -1, -1},
     // target
     {{1, 6, 5, 7, 5}, {3, 8, 6, 9, 1}, {5, 9, 1, 8, 2}}},
    {// dynamic
     {{1, 5}, {1, 64}, {1, 25}, {2, 10}, {1, 64}},
     // target
     {{2, 5, 5, 7, 7}, {1, 4, 10, 2, 11}, {3, 7, 5, 9, 8}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Nspc2NcspSpecial,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes5D_Nspc2NcspSpecial),
                                            ::testing::Values(1),
                                            ::testing::Values(LengthsPerInfer{{2, 1, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(perChannelsToPlanar_5D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_planar_static,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::Values(InputShape{{}, {{3, 6, 5, 6}}}),
                                            ::testing::Values(2, 3),
                                            ::testing::Values(LengthsPerInfer{{1, 3, -1}}),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref, perChannels_4D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_planar = {
    {// dynamic
     {-1, -1, -1, -1},
     // target
     {{1, 9, 8, 7}, {3, 8, 6, 5}, {5, 3, 7, 6}}},
    {// dynamic
     {{1, 5}, {1, 64}, {1, 48}, {2, 48}},
     // target
     {{2, 9, 5, 6}, {1, 6, 9, 8}, {3, 1, 6, 7}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_planar,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_planar),
                                            ::testing::Values(2, 3),
                                            ::testing::Values(LengthsPerInfer{{1, 3, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref, perChannels_4D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_block = {
    {{}, {{3, 16, 6, 7}}},
    {// dynamic
     {-1, 16, -1, -1},
     // target
     {{1, 16, 8, 7}, {3, 16, 7, 8}, {5, 16, 9, 8}}},
    {// dynamic
     {{1, 5}, 16, {1, 48}, {2, 24}},
     // target
     {{2, 16, 12, 6}, {1, 16, 6, 9}, {3, 16, 7, 6}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Block8,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_block),
                                            ::testing::Values(2, 3),
                                            ::testing::Values(LengthsPerInfer{{2, 2, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_4D_ref)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_Block16,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_block),
                                            ::testing::Values(2, 3),
                                            ::testing::Values(LengthsPerInfer{{2, 2, -1, 1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked16_4D_ref)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_planar_static,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::Values(InputShape{{}, {{3, 24, 4, 5, 6}}}),
                                            ::testing::Values(2, 3, 4),
                                            ::testing::Values(LengthsPerInfer{{2, 1, -1}}),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, perChannels_5D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_planar = {
    {// dynamic
     {-1, -1, -1, -1, -1},
     // target
     {{1, 2, 4, 6, 5}, {3, 1, 6, 4, 5}, {5, 6, 5, 7, 4}}},
    {// dynamic
     {{1, 5}, {1, 64}, {1, 48}, {2, 48}, {2, 40}},
     // target
     {{2, 5, 4, 5, 6}, {1, 7, 5, 4, 7}, {3, 3, 5, 6, 4}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_planar,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes5D_planar),
                                            ::testing::Values(2, 3, 4),
                                            ::testing::Values(LengthsPerInfer{{2, 1, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_5D_ref, perChannels_5D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_block = {
    {{}, {{3, 16, 8, 5, 6}}},
    {// dynamic
     {-1, 16, -1, -1, -1},
     // target
     {{1, 16, 5, 6, 7}, {3, 16, 24, 5, 8}, {5, 16, 6, 7, 5}}},
    {// dynamic
     {{1, 5}, 16, {1, 48}, {2, 24}, {2, 64}},
     // target
     {{2, 16, 7, 6, 5}, {1, 16, 6, 5, 7}, {3, 16, 5, 7, 6}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Block8,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes5D_block),
                                            ::testing::Values(2, 3, 4),
                                            ::testing::Values(LengthsPerInfer{{1, 2, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked8_5D_ref)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit5D_CPU_Block16,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes5D_block),
                                            ::testing::Values(2, 3, 4),
                                            ::testing::Values(LengthsPerInfer{{2, 1, -1, 1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(blocked16_5D_ref)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit3D_static,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::Values(InputShape{{}, {{14, 7, 21}}}),
                                            ::testing::Values(1, 2),
                                            ::testing::Values(LengthsPerInfer{{2, 4, -1}}),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes3D = {
    {// dynamic
     {-1, -1, -1},
     // target
     {
         {7, 21, 14},
         {21, 7, 14},
         {21, 14, 7},
     }},
    {// dynamic
     {{1, 60}, {1, 50}, {1, 48}},
     // target
     {
         {14, 21, 7},
         {21, 7, 14},
         {7, 14, 21},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit3D,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes3D),
                                            ::testing::Values(0, 1, 2),
                                            ::testing::Values(LengthsPerInfer{{2, 4, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit2D_static,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::Values(InputShape{{}, {{6, 12}}}),
                                            ::testing::Values(1),
                                            ::testing::Values(LengthsPerInfer{{2, -1}}),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes2D = {
    {// dynamic
     {-1, -1},
     // target
     {
         {3, 8},
         {10, 4},
         {3, 6},
     }},
    {// dynamic
     {{1, 60}, {1, 50}},
     // target
     {
         {3, 4},
         {4, 4},
         {6, 12},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit2D,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes2D),
                                            ::testing::Values(0, 1),
                                            ::testing::Values(LengthsPerInfer{{2, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit1D_static,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::Values(InputShape{{}, {{10}}}),
                                            ::testing::Values(0),
                                            ::testing::Values(LengthsPerInfer{{2, 1, 1, -1}}),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"},
                                                              CPUSpecificParams{{}, {}, {"ref"}, "ref"})),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes1D = {
    {// dynamic
     {-1},
     // target
     {
         {5},
         {15},
         {10},
     }},
    {// dynamic
     {{1, 60}},
     // target
     {
         {15},
         {5},
         {10},
     }},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit1D,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes1D),
                                            ::testing::Values(0),
                                            ::testing::Values(LengthsPerInfer{{2, 1, 1, -1}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref"})),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_zero_dims = {{// dynamic
                                                          {-1, -1, -1, -1},
                                                          // target
                                                          {
                                                              {1, 7, 7, 7},
                                                              {3, 7, 7, 7},
                                                          }}};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_zero_dims,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_zero_dims),
                                            ::testing::Values(1, 2, 3),
                                            ::testing::Values(LengthsPerInfer{{3, 4, -1}},
                                                              LengthsPerInfer{{3, -1, 4}},
                                                              LengthsPerInfer{{-1, 3, 4}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(planar_4D_ref)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_zero_dims_nspc_ncsp,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_zero_dims),
                                            ::testing::Values(1),
                                            ::testing::Values(LengthsPerInfer{{3, 4, -1}},
                                                              LengthsPerInfer{{3, -1, 4}},
                                                              LengthsPerInfer{{-1, 3, 4}}),
                                            ::testing::ValuesIn(lengthsTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(perChannelsToPlanar_4D)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes4D_dynamic_lengths = {
    {{1, 16, 8, 7}, {{1, 16, 8, 7}, {1, 16, 8, 7}, {1, 16, 8, 7}}},
    {{-1, -1, -1, -1}, {{1, 16, 8, 7}, {1, 16, 8, 7}, {1, 16, 8, 7}}},
    {{{1, 5}, -1, {1, 48}, {2, 24}}, {{2, 16, 12, 6}, {1, 16, 6, 9}, {3, 16, 7, 6}}},
};

std::vector<LengthsPerInfer> lengthsPerInfer = {
    LengthsPerInfer{{10, 4, 2}, {10, 4, 2}, {10, 4, 2}},
    LengthsPerInfer{{10, 4, 2}, {10, 4, 2}, {5, 5, 6}},
    LengthsPerInfer{{10, 4, 2}, {2, 4, 10}, {4, 2, 10}},
};

INSTANTIATE_TEST_SUITE_P(smoke_VariadicSplit4D_CPU_dynamic_lengths,
                         VariadicSplitLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes4D_dynamic_lengths),
                                            ::testing::Values(1),
                                            ::testing::ValuesIn(lengthsPerInfer),
                                            ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(planar_4D_ref)),
                         VariadicSplitLayerCPUTest::getTestCaseName);

// =========================================== in - place ============================================================//
INSTANTIATE_TEST_SUITE_P(
    smoke_VariadicSplit_CPU_planar_inPlace_0,
    VariadicSplitLayerCPUTest,
    ::testing::Combine(::testing::Values(InputShape{{}, {{5, 6, 5, 6, 7}}},
                                         InputShape{{}, {{5, 6, 5, 6}}},
                                         InputShape{{}, {{5, 6, 5}}},
                                         InputShape{{5, -1, -1, -1, -1},
                                                    {{5, 6, 5, 6, 7}, {5, 2, 5, 2, 7}, {5, 8, 5, 8, 7}}},
                                         InputShape{{5, -1, -1, -1}, {{5, 6, 5, 6}, {5, 2, 5, 2}, {5, 8, 5, 8}}},
                                         InputShape{{5, -1, -1}, {{5, 6, 5}, {5, 2, 5}, {5, 8, 5}}}),
                       ::testing::Values(0),
                       ::testing::Values(LengthsPerInfer{{1, 2, -1}}),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
    VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_VariadicSplit_CPU_planar_inPlace_1,
    VariadicSplitLayerCPUTest,
    ::testing::Combine(::testing::Values(InputShape{{}, {{1, 6, 5, 6, 7}}},
                                         InputShape{{}, {{1, 6, 5, 6}}},
                                         InputShape{{}, {{1, 6, 5}}},
                                         InputShape{{1, 6, -1, -1, -1},
                                                    {{1, 6, 5, 6, 7}, {1, 6, 5, 2, 7}, {1, 6, 5, 8, 7}}},
                                         InputShape{{1, 6, -1, -1}, {{1, 6, 5, 6}, {1, 6, 5, 2}, {1, 6, 5, 8}}},
                                         InputShape{{1, 6, -1}, {{1, 6, 5}, {1, 6, 3}, {1, 6, 7}}}),
                       ::testing::Values(1),
                       ::testing::Values(LengthsPerInfer{{1, 2, -1}}),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(CPUSpecificParams{{}, {}, {}, "unknown"})),
    VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_VariadicSplit4D_CPU_block8_inPlace,
    VariadicSplitLayerCPUTest,
    ::testing::Combine(::testing::Values(InputShape{{}, {{1, 32, 5, 6}}},
                                         InputShape{{1, 32, -1, -1}, {{1, 32, 5, 6}, {1, 32, 5, 2}, {1, 32, 5, 8}}}),
                       ::testing::Values(1),
                       ::testing::Values(LengthsPerInfer{{8, 16, -1}}),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(blocked8_4D)),
    VariadicSplitLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_VariadicSplit4D_CPU_block16_inPlace,
    VariadicSplitLayerCPUTest,
    ::testing::Combine(::testing::Values(InputShape{{}, {{1, 64, 5, 6}}},
                                         InputShape{{1, 64, -1, -1}, {{1, 64, 5, 6}, {1, 64, 5, 2}, {1, 64, 5, 8}}}),
                       ::testing::Values(1),
                       ::testing::Values(LengthsPerInfer{{16, 32, -1}}),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(blocked16_4D)),
    VariadicSplitLayerCPUTest::getTestCaseName);
}  // namespace