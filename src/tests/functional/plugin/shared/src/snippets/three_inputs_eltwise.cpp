// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/three_inputs_eltwise.hpp"
#include "subgraph_simple.hpp"

namespace LayerTestsDefinitions {

    std::string ThreeInputsEltwise::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShapes0, inputShapes1, inputShapes2;
        std::string targetDevice;
        size_t num_nodes, num_subgraphs;
        std::tie(netPrecision, inputShapes0, inputShapes1, inputShapes2,
                 num_nodes, num_subgraphs, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShapes0) << "_";
        result << "IS[1]=" << CommonTestUtils::vec2str(inputShapes1) << "_";
        result << "IS[2]=" << CommonTestUtils::vec2str(inputShapes2) << "_";
        result << "#N=" << num_nodes << "_";
        result << "#S=" << num_subgraphs << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    void ThreeInputsEltwise::SetUp() {
        std::vector<size_t> inputShape0, inputShape1, inputShape2;
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, inputShape0, inputShape1, inputShape2,
                 ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
        init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}, {{}, {inputShape2, }}});

        auto f = ov::test::snippets::EltwiseFunctionThreeInputs({inputShape0, inputShape1, inputShape2});
        function = f.getOriginal();
    }

    void ThreeInputsEltwiseConvert::SetUp() {
        std::vector<size_t> inputShape0, inputShape1, inputShape2;
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, inputShape0, inputShape1, inputShape2,
                 ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();
        init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}, {{}, {inputShape2, }}});

        auto f = ov::test::snippets::EltwiseFunctionThreeInputsConvert({inputShape0, inputShape1, inputShape2});
        function = f.getOriginal();
    }

TEST_P(ThreeInputsEltwise, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

TEST_P(ThreeInputsEltwiseConvert, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}
}  // namespace LayerTestsDefinitions
