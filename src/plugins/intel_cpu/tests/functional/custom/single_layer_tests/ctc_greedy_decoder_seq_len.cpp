// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>
#include <gtest/gtest.h>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

using CtcGreedyDecoderSeqLenParams = std::tuple<size_t,   // Batch size N
                                                size_t,   // Sequence length T
                                                size_t>;  // Number of classes

using InputShapeParams = std::pair<std::vector<ov::Dimension>,                  // bounds N, T, C, blank
                                   std::vector<CtcGreedyDecoderSeqLenParams>>;  // target input dimensions

using InputElementParams = std::vector<ElementType>;

using CTCGreedyDecoderSeqLenLayerCPUTestParams = std::tuple<InputShapeParams,    // Input Shape
                                                            InputElementParams,  // Input precision
                                                            ElementType,         // Index Type
                                                            bool                 // mergeRepeated
                                                            >;

class CTCGreedyDecoderSeqLenLayerCPUTest : public testing::WithParamInterface<CTCGreedyDecoderSeqLenLayerCPUTestParams>,
                                           virtual public SubgraphBaseTest,
                                           public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCGreedyDecoderSeqLenLayerCPUTestParams>& obj) {
        InputElementParams inType;
        bool mergeRepeated;
        InputShapeParams shapes;
        ElementType indexType;
        std::tie(shapes, inType, indexType, mergeRepeated) = obj.param;
        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& shape : shapes.second) {
            size_t N;
            size_t T;
            size_t C;
            std::tie(N, T, C) = shape;
            results << "{" << N << "," << T << "," << C << "}"
                    << "_";
        }
        for (const auto& type : inType) {
            results << "Prc=" << type << "_";
        }

        results << "IndexType=" << indexType << "_";

        results << "MergeRepeated=" << mergeRepeated;

        return results.str();
    }

protected:
    void SetUp() override {
        InputElementParams inType;
        bool mergeRepeated;
        InputShapeParams shapes;
        ElementType indexType;
        std::tie(shapes, inType, indexType, mergeRepeated) = GetParam();
        selectedType = "ref_any_f32";
        targetDevice = ov::test::utils::DEVICE_CPU;
        ASSERT_EQ(shapes.first.size(), 4);
        const auto& in_dyn_N = shapes.first[0];
        const auto& in_dyn_T = shapes.first[1];
        const auto& in_dyc_C = shapes.first[2];
        const auto& in_dyc_blank = shapes.first[3];
        const size_t blank_rank = in_dyc_blank.get_length();
        ASSERT_TRUE(blank_rank == 0 || blank_rank == 1);
        inputDynamicShapes = {ov::PartialShape{in_dyn_N, in_dyn_T, in_dyc_C},
                              ov::PartialShape{in_dyn_N},
                              blank_rank == 0 ? ov::PartialShape{} : ov::PartialShape{1}};
        OPENVINO_ASSERT(inType.size() == inputDynamicShapes.size());

        for (auto& shape : shapes.second) {
            size_t N;
            size_t T;
            size_t C;
            std::tie(N, T, C) = shape;
            if (blank_rank == 0)
                targetStaticShapes.push_back({{N, T, C}, {N}, {}});
            else
                targetStaticShapes.push_back({{N, T, C}, {N}, {1}});
        }

        ov::ParameterVector params;
        for (size_t i = 0; i < inType.size(); i++) {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(inType[i], inputDynamicShapes[i]);
            params.push_back(param_node);
        }
        auto ctcGreedyDecoderSeqLen = std::make_shared<ov::op::v6::CTCGreedyDecoderSeqLen>(params[0],
                                                                                           params[1],
                                                                                           params[2],
                                                                                           mergeRepeated,
                                                                                           indexType,
                                                                                           indexType);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(ctcGreedyDecoderSeqLen)};
        function = std::make_shared<ov::Model>(results, params, "CTCGreedyDecoderSeqLenCPU");
    };

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        const auto& dataShape = targetInputStaticShapes[0];
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                if (funcInput.get_element_type().is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            } else if (i == 1) {
                const auto seqLen = dataShape[1];
                const auto B = dataShape[0];
                std::mt19937 gen(42);
                std::uniform_int_distribution<unsigned long> dist(1, seqLen);

                std::vector<int32_t> sequenceLenData(B, 0);
                for (size_t b = 0; b < B; b++) {
                    const int len = dist(gen);
                    sequenceLenData[b] = len;
                }
                tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                if (funcInput.get_element_type() == ElementType::i32) {
                    auto begin = tensor.data<int32_t>();
                    std::copy(sequenceLenData.begin(), sequenceLenData.end(), begin);
                } else if (funcInput.get_element_type() == ElementType::i64) {
                    auto begin = tensor.data<int64_t>();
                    std::copy(sequenceLenData.begin(), sequenceLenData.end(), begin);
                }

            } else if (i == 2) {
                // blank should be valid class type
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = dataShape[2];
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(CTCGreedyDecoderSeqLenLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "CTCGreedyDecoderSeqLen");
}

namespace {
// Common params
// ElementType::f16 is not support by CPU Plugin yet
const std::vector<ElementType> inputType = {ElementType::f32, ElementType::i32, ElementType::i32};
const std::vector<bool> mergeRepeated{true, false};
const std::vector<ElementType> indexType = {ElementType::i64, ElementType::i32};
const std::vector<InputShapeParams> inputShapesCTCDecoder = {
    {{ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{0}},
     {CtcGreedyDecoderSeqLenParams{1, 1, 1},
      CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
    {{ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{1}},
     {CtcGreedyDecoderSeqLenParams{1, 1, 1},
      CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
    {{ov::Dimension{1, 5}, ov::Dimension{1, 6}, ov::Dimension{1, 60}, ov::Dimension{0}},
     {CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
    {{ov::Dimension{1, 5}, ov::Dimension{1, 6}, ov::Dimension{1, 60}, ov::Dimension{1}},
     {CtcGreedyDecoderSeqLenParams{1, 6, 10},
      CtcGreedyDecoderSeqLenParams{3, 3, 16},
      CtcGreedyDecoderSeqLenParams{5, 3, 55}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inputShapesCTCDecoder),
                                           ::testing::Values(inputType),
                                           ::testing::ValuesIn(indexType),
                                           ::testing::ValuesIn(mergeRepeated));

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderSeqLenCPU,
                         CTCGreedyDecoderSeqLenLayerCPUTest,
                         basicCases,
                         CTCGreedyDecoderSeqLenLayerCPUTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
