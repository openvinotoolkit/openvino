// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/common_utils.hpp"

#include <algorithm>
#include <cassert>

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using ConvertFqRnnToQuantizedRnnTestParams = std::tuple<std::string, SizeVector>;
/* using ConvertFqRnnToQuantizedRnnTestParams = std::string; */

class ConvertFqRnnToQuantizedRnn : public testing::WithParamInterface<ConvertFqRnnToQuantizedRnnTestParams>,
                                   public CpuTestWithFusing,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertFqRnnToQuantizedRnnTestParams>& obj) {
        SizeVector inputShapes;
        std::string rnnType;
        std::tie(rnnType, inputShapes) = obj.param;

        auto batchSize  = inputShapes[0];
        auto inputSize  = inputShapes[1];
        auto hiddenSize = inputShapes[2];

        std::ostringstream result;
        result << "Type = " << rnnType << "_";
        result << "batch = " << batchSize << "_";
        result << "input = " << inputSize << "_";
        result << "hidden = " << hiddenSize << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        SizeVector inputShapes;
        std::string rnnType;

        std::tie(rnnType, inputShapes) = this->GetParam();

        auto batchSize  = inputShapes[0];
        auto inputSize  = inputShapes[1];
        auto hiddenSize = inputShapes[2];

        const float inputDataMin = 6.43123;
        const float inputDataMax = -6.48187;
        const float outputDataMin = inputDataMin;
        const float outputDataMax = outputDataMin;

        const SizeVector inputShape       = {batchSize, inputSize};
        const SizeVector hiddenStateShape = {batchSize, hiddenSize};
        const SizeVector cellStateShape   = {batchSize, hiddenSize};

        init_input_shapes({
                {{}, {inputShape}},
                {{}, {hiddenStateShape}},
                {{}, {cellStateShape}}
            });

        const auto ngPrec = element::f32;
        auto inputParams = builder::makeParams(ngPrec, {inputShape, hiddenStateShape, cellStateShape});
        const auto outputNodes = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

        std::vector<float> empty;
        auto W = ngraph::builder::makeConstant(ngraph::element::f32, {4 * hiddenSize, inputSize},  empty, true);
        auto R = ngraph::builder::makeConstant(ngraph::element::f32, {4 * hiddenSize, hiddenSize}, empty, true);
        auto B = ngraph::builder::makeConstant(ngraph::element::f32, {4 * hiddenSize},             empty, true);

        const auto fqLevels = 256;

        auto inputFQ = ngraph::builder::makeFakeQuantize(outputNodes[0], ngraph::element::f32, fqLevels, std::vector<size_t>{},
                                                         { inputDataMin }, { inputDataMax }, { outputDataMin }, { outputDataMax });

        auto hiddenStateFQ = ngraph::builder::makeFakeQuantize(outputNodes[1], ngraph::element::f32, fqLevels, std::vector<size_t>{},
                                                             { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });

        auto weightsFQ = ngraph::builder::makeFakeQuantize(W, ngraph::element::f32, fqLevels, std::vector<size_t>{},
                                                             { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });

        auto recurrentWeightsFQ = ngraph::builder::makeFakeQuantize(R, ngraph::element::f32, fqLevels, std::vector<size_t>{},
                                                               { inputDataMin }, { inputDataMax }, { inputDataMin }, { inputDataMax });

        auto rnnCellOp = std::make_shared<ov::op::v4::LSTMCell>(inputFQ, hiddenStateFQ, inputParams[2], weightsFQ, recurrentWeightsFQ, B, hiddenSize);

        function = makeNgraphFunction(ngPrec, inputParams, rnnCellOp, "ConvertFqRnnToQuantizedRnn");
    }
};

TEST_P(ConvertFqRnnToQuantizedRnn, CompareWithRefs) {
    run();
}

namespace {

const std::vector<SizeVector> inputShapes {
    {37, 128, 512},
    /* {256, 128, 256}, */
};

std::vector<std::string> rnnTypes {"LSTMCell", "RNNCell", "GRUCell"};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ConvertFqRnnToQuantizedRnn,
                         /* ::testing::ValuesIn(rnnTypes), */
                         ::testing::Combine(::testing::ValuesIn(rnnTypes),
                                            ::testing::ValuesIn(inputShapes)),
                         ConvertFqRnnToQuantizedRnn::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
