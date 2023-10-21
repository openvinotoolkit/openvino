// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <ov_models/builders.hpp>
#include <shared_test_classes/single_layer/ctc_greedy_decoder.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

using CtcGreedyDecoderParams = std::tuple<size_t,   // Sequence length T
                                          size_t,   // Batch size N
                                          size_t>;  // Number of classes

using InputShapeParams = std::pair<std::vector<ov::Dimension>,            // bounds for T, N, C
                                   std::vector<CtcGreedyDecoderParams>>;  // target input dimensions

using CTCGreedyDecoderLayerCPUTestParams = std::tuple<InputShapeParams,  // Input Shape
                                                      ElementType,       // Input precision
                                                      bool               // mergeRepeated
                                                      >;

class CTCGreedyDecoderLayerCPUTest : public testing::WithParamInterface<CTCGreedyDecoderLayerCPUTestParams>,
                                     virtual public SubgraphBaseTest,
                                     public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCGreedyDecoderLayerCPUTestParams>& obj) {
        ElementType inType;
        bool mergeRepeated;
        InputShapeParams shapes;
        std::tie(shapes, inType, mergeRepeated) = obj.param;
        std::ostringstream results;
        results << "IS=" << ov::test::utils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& shape : shapes.second) {
            size_t T;
            size_t N;
            size_t C;
            std::tie(T, N, C) = shape;
            results << "{" << T << "," << N << "," << C << "}"
                    << "_";
        }

        results << "Prc=" << inType << "_";
        results << "MergeRepeated=" << mergeRepeated;

        return results.str();
    }

protected:
    void SetUp() override {
        ElementType inType;
        bool mergeRepeated;
        InputShapeParams shapes;
        std::tie(shapes, inType, mergeRepeated) = GetParam();
        selectedType = "ref_any_FP32";
        targetDevice = ov::test::utils::DEVICE_CPU;
        // construct input shapes
        ASSERT_EQ(shapes.first.size(), 3);
        const auto& in_dyn_T = shapes.first[0];
        const auto& in_dyn_N = shapes.first[1];
        const auto& in_dyc_C = shapes.first[2];
        inputDynamicShapes = {ov::PartialShape{in_dyn_T, in_dyn_N, in_dyc_C}, ov::PartialShape{in_dyn_T, in_dyn_N}};

        for (const auto& shape : shapes.second) {
            size_t T;
            size_t N;
            size_t C;
            std::tie(T, N, C) = shape;
            targetStaticShapes.push_back({{T, N, C}, {T, N}});
        }

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto ctcGreedyDecoder = std::make_shared<ov::op::v0::CTCGreedyDecoder>(params[0], params[1], mergeRepeated);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoder)};
        function = std::make_shared<ngraph::Function>(results, params, "CTCGreedyDecoderCPU");
    };

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 0) {
                if (funcInput.get_element_type().is_real()) {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                     targetInputStaticShapes[i],
                                                                     10,
                                                                     0,
                                                                     1000);
                } else {
                    tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                     targetInputStaticShapes[i]);
                }
            } else {
                auto T = targetInputStaticShapes[i][0];
                auto B = targetInputStaticShapes[i][1];
                std::mt19937 gen(1);
                std::uniform_int_distribution<unsigned long> dist(1, T);

                std::vector<float> sequenceMaskData(B * T, 0);
                for (size_t b = 0; b < B; b++) {
                    int len = dist(gen);
                    for (int t = 0; t < len; t++) {
                        sequenceMaskData[t * B + b] = 1;
                    }
                }
                tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                float* begin = tensor.data<float>();
                std::copy(sequenceMaskData.begin(), sequenceMaskData.end(), begin);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(CTCGreedyDecoderLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "CTCGreedyDecoder");
}

namespace {
// Common params
// ElementType::f16 is not support by CPU Plugin yet
const std::vector<ElementType> netPrecisions = {ElementType::f32};
const std::vector<bool> mergeRepeated{true, false};
const std::vector<InputShapeParams> inputShapesCTCDecoder = {
    {{ov::Dimension{1, 50}, ov::Dimension{1, 3}, ov::Dimension{2, 150}},
     {CtcGreedyDecoderParams{1, 1, 16},
      CtcGreedyDecoderParams{50, 3, 3},
      CtcGreedyDecoderParams{50, 3, 7},
      CtcGreedyDecoderParams{50, 3, 8},
      CtcGreedyDecoderParams{50, 3, 16},
      CtcGreedyDecoderParams{50, 3, 128},
      CtcGreedyDecoderParams{50, 3, 49},
      CtcGreedyDecoderParams{50, 3, 55}}},
    {{ov::Dimension{-1}, ov::Dimension{-1}, ov::Dimension{-1}},
     {CtcGreedyDecoderParams{50, 3, 3},
      CtcGreedyDecoderParams{50, 3, 7},
      CtcGreedyDecoderParams{50, 3, 8},
      CtcGreedyDecoderParams{50, 3, 16},
      CtcGreedyDecoderParams{50, 3, 128},
      CtcGreedyDecoderParams{50, 3, 49},
      CtcGreedyDecoderParams{50, 3, 55},
      CtcGreedyDecoderParams{1, 1, 16}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inputShapesCTCDecoder),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(mergeRepeated));

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderCPU,
                         CTCGreedyDecoderLayerCPUTest,
                         basicCases,
                         CTCGreedyDecoderLayerCPUTest::getTestCaseName);
}  // namespace

}  // namespace CPULayerTestsDefinitions
