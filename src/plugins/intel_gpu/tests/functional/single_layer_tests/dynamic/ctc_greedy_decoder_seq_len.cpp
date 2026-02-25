// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <random>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        InputShape,                // Input shape
        int,                       // Sequence lengths
        ov::element::Type,         // Probabilities precision
        ov::element::Type,         // Indices precision
        int,                       // Blank index
        bool,                      // Merge repeated
        std::string                // Device name
> ctcGreedyDecoderSeqLenParams;

class CTCGreedyDecoderSeqLenLayerGPUTest
    : public testing::WithParamInterface<ctcGreedyDecoderSeqLenParams>,
      virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ctcGreedyDecoderSeqLenParams>& obj) {
        const auto& [inputShape, sequenceLengths, dataPrecision, indicesPrecision, blankIndex, mergeRepeated, targetDevice] = obj.param;

        std::ostringstream result;

        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_" << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }
        result << ")_";
        result << "seqLen=" << sequenceLengths << '_';
        result << "dataPRC=" << dataPrecision.get_type_name() << '_';
        result << "idxPRC=" << indicesPrecision.get_type_name() << '_';
        result << "BlankIdx=" << blankIndex << '_';
        result << "mergeRepeated=" << std::boolalpha << mergeRepeated << '_';
        result << "trgDev=" << targetDevice;

        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [inputShape, sequenceLengths, model_type, indices_type, _blankIndex, mergeRepeated, _targetDevice] = GetParam();
        auto blankIndex = _blankIndex;
        targetDevice = _targetDevice;
        inputDynamicShapes = {inputShape.first, {}};
        for (size_t i = 0; i < inputShape.second.size(); ++i) {
            targetStaticShapes.push_back({inputShape.second[i], {}});
        }

        ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front())};

        const auto sequenceLenNode = [&, sequenceLengths = sequenceLengths, indices_type = indices_type] {
            const size_t B = targetStaticShapes[0][0][0];
            const size_t T = targetStaticShapes[0][0][1];

            // Cap sequence length up to T
            const int seqLen = std::min<int>(T, sequenceLengths);

            std::mt19937 gen{42};
            std::uniform_int_distribution<int> dist(1, seqLen);

            std::vector<int> sequenceLenData(B);
            for (size_t b = 0; b < B; b++) {
                const int len = dist(gen);
                sequenceLenData[b] = len;
            }

            return std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{B}, sequenceLenData);
        }();

        // Cap blank index up to C - 1
        int C = targetStaticShapes[0][0][2];
        blankIndex = std::min(blankIndex, C - 1);

        const auto blankIndexNode = [&, indices_type = indices_type] {
            if (indices_type == ov::element::i32) {
                const auto blankIdxDataI32 = std::vector<int32_t>{blankIndex};
                return std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{1}, blankIdxDataI32);
            } else if (indices_type == ov::element::i64) {
                const auto blankIdxDataI64 = std::vector<int64_t>{blankIndex};
                return std::make_shared<ov::op::v0::Constant>(indices_type, ov::Shape{1}, blankIdxDataI64);
            }
            throw std::logic_error("Unsupported index precision");
        }();

        auto ctcGreedyDecoderSeqLen = std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(params[0],
                                                                                           sequenceLenNode,
                                                                                           blankIndexNode,
                                                                                           mergeRepeated,
                                                                                           indices_type,
                                                                                           indices_type);

        ov::OutputVector results;
        for (size_t i = 0; i < ctcGreedyDecoderSeqLen->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(ctcGreedyDecoderSeqLen->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "CTCGreedyDecoderSeqLen");
    }
};

TEST_P(CTCGreedyDecoderSeqLenLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
};

namespace {

std::vector<ov::test::InputShape> inputShapeDynamic = {
    {
        {{-1, -1, -1}, {{1, 28, 41}}},
        {{-1, -1, -1}, {{1, 1, 1}}},
        {{-1, -1, -1}, {{1, 6, 10}}},
        {{-1, -1, -1}, {{3, 3, 16}}},
        {{-1, -1, -1}, {{5, 3, 55}}},
    }
};

const std::vector<ov::element::Type> probPrecisions = {
    ov::element::f32,
    ov::element::f16
};
const std::vector<ov::element::Type> idxPrecisions = {
    ov::element::i32,
    ov::element::i64
};

std::vector<bool> mergeRepeated{true, false};

INSTANTIATE_TEST_SUITE_P(smoke_ctc_greedy_decoder_seq_len_dynamic,
                         CTCGreedyDecoderSeqLenLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapeDynamic),
                                            ::testing::Values(10),
                                            ::testing::ValuesIn(probPrecisions),
                                            ::testing::ValuesIn(idxPrecisions),
                                            ::testing::Values(0),
                                            ::testing::ValuesIn(mergeRepeated),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         CTCGreedyDecoderSeqLenLayerGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ctc_greedy_decoder_seq_len_bi_dynamic,
                         CTCGreedyDecoderSeqLenLayerGPUTest,
                         ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::InputShape>{
                                                {{-1, -1, -1}, {{2, 8, 11}}},
                                                {{-1, -1, -1}, {{4, 10, 55}}}}),
                                            ::testing::ValuesIn(std::vector<int>{5, 100}),
                                            ::testing::ValuesIn(probPrecisions),
                                            ::testing::ValuesIn(idxPrecisions),
                                            ::testing::ValuesIn(std::vector<int>{0, 5, 10}),
                                            ::testing::ValuesIn(mergeRepeated),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         CTCGreedyDecoderSeqLenLayerGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
