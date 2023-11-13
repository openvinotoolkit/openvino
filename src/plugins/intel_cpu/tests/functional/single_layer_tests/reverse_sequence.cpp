// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using ReverseSequenceCPUTestParams = typename std::tuple<
        int64_t,                              // Index of the batch dimension
        int64_t,                              // Index of the sequence dimension
        InputShape,                           // Input shape
        InputShape,                           // Shape of the input vector with sequence lengths to be reversed
        ngraph::helpers::InputLayerType,      // Secondary input type
        InferenceEngine::Precision,           // Network precision
        std::string>;                         // Device name

class ReverseSequenceLayerCPUTest : public testing::WithParamInterface<ReverseSequenceCPUTestParams>,
                                    virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ReverseSequenceCPUTestParams> obj) {
        int64_t batchAxisIndex;
        int64_t seqAxisIndex;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        InputShape dataInputShape;
        InputShape seqLengthsShape;
        ngraph::helpers::InputLayerType secondaryInputType;

        std::tie(batchAxisIndex, seqAxisIndex, dataInputShape, seqLengthsShape, secondaryInputType, netPrecision, targetName) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({dataInputShape.first}) << "_";
        result << "TS=";
        for (const auto& item : dataInputShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "seqLengthsShape" << ov::test::utils::partialShape2str({seqLengthsShape.first}) << "_";
        result << "seqLengthsShapes=";
        for (const auto& item : seqLengthsShape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "secondaryInputType=" << secondaryInputType << "_";
        result << "batchAxis=" << batchAxisIndex << "_";
        result << "seqAxis=" << seqAxisIndex << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetName;

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        int64_t batchAxisIndex;
        int64_t seqAxisIndex;
        InputShape dataInputShape;
        InputShape seqLengthsShape;
        ngraph::helpers::InputLayerType secondaryInputType;

        std::tie(batchAxisIndex, seqAxisIndex, dataInputShape, seqLengthsShape, secondaryInputType, netPrecision, targetDevice) = GetParam();
        m_seqAxisIndex = seqAxisIndex;

        init_input_shapes({dataInputShape, seqLengthsShape});

        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(ngPrc, inputDynamicShapes[0])};

        constexpr auto seqLengthsPrc = ngraph::element::Type_t::i32; //according to the specification
        std::shared_ptr<ngraph::Node> seqLengthsInput;

        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            seqLengthsInput = ngraph::builder::makeDynamicInputLayer(seqLengthsPrc, secondaryInputType, inputDynamicShapes[1]);
            paramsIn.push_back(std::dynamic_pointer_cast<ngraph::opset3::Parameter>(seqLengthsInput));
        } else {
            const auto maxSeqLength = dataInputShape.second.front().at(seqAxisIndex);
            seqLengthsInput = ngraph::builder::makeConstant<float>(seqLengthsPrc, seqLengthsShape.second.front(), {}, true, maxSeqLength);
        }

        const auto reverse = std::make_shared<ngraph::opset1::ReverseSequence>(paramsIn.front(), seqLengthsInput, batchAxisIndex, seqAxisIndex);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reverse)};
        function = std::make_shared<ngraph::Function>(results, paramsIn, "ReverseSequence");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const auto dataInputTensor =
            ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                    targetInputStaticShapes[0]);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), dataInputTensor});

        if (funcInputs.size() != 1) {
            const auto maxSeqLength = targetInputStaticShapes.front().at(m_seqAxisIndex);
            const auto seqLengthsTensor =
                ov::test::utils::create_and_fill_tensor(funcInputs[1].get_element_type(),
                                                        targetInputStaticShapes[1],
                                                        maxSeqLength, 1);
            inputs.insert({funcInputs[1].get_node_shared_ptr(), seqLengthsTensor});
        }
    }

private:
    int64_t m_seqAxisIndex;
};

TEST_P(ReverseSequenceLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
};

const int64_t batchAxisIndex = 0L;

const std::vector<int64_t> seqAxisIndices = {1L, 2L};

const std::vector<InputShape> dataInputStaticShapes3D = {{{}, {{3, 10, 20}}}, {{}, {{3, 12, 48}}}};

const std::vector<InputShape> dataInputStaticShapes4D = {{{}, {{3, 7, 10, 20}}}, {{}, {{3, 12, 5, 4}}}};

const std::vector<InputShape> dataInputStaticShapes5D = {{{}, {{3, 12, 7, 10, 2}}}, {{}, {{3, 3, 12, 1, 40}}}};

const std::vector<InputShape> seqLengthsStaticShapes = {{{}, {{3}}}};

const std::vector<InputShape> dataInputDynamicShapes3D =
    {{{-1, -1, {5, 10}}, {{7, 20, 8}, {10, 15, 10}}}, {{-1, -1, -1}, {{7, 4, 1}, {10, 42, 70}}}};

const std::vector<InputShape> dataInputDynamicShapes4D =
    {{{-1, -1, {5, 10}, -1}, {{7, 20, 8, 4}, {10, 2, 7, 50}}}, {{-1, -1, -1, -1}, {{7, 15, 1, 100}, {10, 4, 10, 12}}}};

const std::vector<InputShape> dataInputDynamicShapes5D =
    {{{-1, -1, {5, 10}, -1, {2, 14}}, {{7, 3, 8, 20, 9}, {10, 12, 10, 3, 2}}},
    {{-1, -1, -1, -1, -1}, {{7, 15, 15, 10, 3}, {10, 4, 100, 90, 5}}}};

const std::vector<InputShape> seqLengthsDynamicShapes = {{{-1}, {{7}, {10}}}};

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequenceCPUStatic3D, ReverseSequenceLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Values(batchAxisIndex),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(dataInputStaticShapes3D),
                            ::testing::ValuesIn(seqLengthsStaticShapes),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequenceCPUStatic4D, ReverseSequenceLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Values(batchAxisIndex),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(dataInputStaticShapes4D),
                            ::testing::ValuesIn(seqLengthsStaticShapes),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequenceCPUStatic5D, ReverseSequenceLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Values(batchAxisIndex),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(dataInputStaticShapes5D),
                            ::testing::ValuesIn(seqLengthsStaticShapes),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequenceCPUDynamic3D, ReverseSequenceLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Values(batchAxisIndex),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(dataInputDynamicShapes3D),
                            ::testing::ValuesIn(seqLengthsDynamicShapes),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequenceCPUDynamic4D, ReverseSequenceLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Values(batchAxisIndex),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(dataInputDynamicShapes4D),
                            ::testing::ValuesIn(seqLengthsDynamicShapes),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequenceCPUDynamic5D, ReverseSequenceLayerCPUTest,
                        ::testing::Combine(
                            ::testing::Values(batchAxisIndex),
                            ::testing::ValuesIn(seqAxisIndices),
                            ::testing::ValuesIn(dataInputDynamicShapes5D),
                            ::testing::ValuesIn(seqLengthsDynamicShapes),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        ReverseSequenceLayerCPUTest::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions
