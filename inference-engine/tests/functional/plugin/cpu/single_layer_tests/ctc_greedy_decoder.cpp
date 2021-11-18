// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional_test_utils/ov_tensor_utils.hpp>
#include <ngraph_functions/builders.hpp>
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

using TargetShapeParams = std::tuple<size_t,   // Sequence length T
                                     size_t,   // Batch size N
                                     size_t>;  // Number of classes

using InputShapeParams = std::pair<std::vector<ov::PartialShape>,    // bounds for input dynamic shape
                                   std::vector<TargetShapeParams>>;  // target input dimensions

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
        results << "IS=" << CommonTestUtils::partialShape2str({shapes.first}) << "_";
        results << "TS=";
        for (const auto& shape : shapes.second) {
            size_t T;
            size_t N;
            size_t C;
            std::tie(T, N, C) = shape;
            results << "{" << T  << "," << N << "," << C  << "}"<< "_";
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

        targetDevice = CommonTestUtils::DEVICE_CPU;
        inputDynamicShapes = shapes.first;

        for (auto& shape : shapes.second) {
            size_t T;
            size_t N;
            size_t C;
            std::tie(T, N, C) = shape;
            targetStaticShapes.push_back({{T, N, C}, {T, N}});
        }

        auto params = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto ctcGreedyDecoder = std::make_shared<ov::op::v0::CTCGreedyDecoder>(params[0], params[1], mergeRepeated);

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(ctcGreedyDecoder)};
        function = std::make_shared<ngraph::Function>(results, params, "CTCGreedyDecoderCPU");
    };

    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;
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
                for (int b = 0; b < B; b++) {
                    int len = dist(gen);
                    for (int t = 0; t < len; t++) {
                        sequenceMaskData[t * B + b] = 1;
                    }
                }
                tensor = ov::runtime::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
                float* begin = tensor.data<float>();
                std::copy(sequenceMaskData.begin(), sequenceMaskData.end(), begin);
            }

            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(CTCGreedyDecoderLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
}

namespace {
// Common params
// ElementType::f16 is not support by CPU Plugin yet
const std::vector<ElementType> netPrecisions = {ElementType::f32};
const std::vector<bool> mergeRepeated{true, false};
const std::vector<InputShapeParams> inputShapesCTCDecoder = {
    {{{-1, -1, -1}, {-1, -1}},
     {{50, 3, 3}, {50, 3, 7}, {50, 3, 8}, {50, 3, 16}, {50, 3, 128}, {50, 3, 49}, {50, 3, 55}, {1, 1, 16}}}};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inputShapesCTCDecoder),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(mergeRepeated));

INSTANTIATE_TEST_SUITE_P(smoke_CtcGreedyDecoderCPU,
                         CTCGreedyDecoderLayerCPUTest,
                         basicCases,
                         CTCGreedyDecoderLayerCPUTest::getTestCaseName);
}  // namespace

}  // namespace CPULayerTestsDefinitions
