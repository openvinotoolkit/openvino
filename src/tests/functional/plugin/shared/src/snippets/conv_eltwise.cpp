// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "snippets/conv_eltwise.hpp"
#include "subgraph_customizable.hpp"

namespace LayerTestsDefinitions {

    std::string ConvEltwise::getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj) {
        InferenceEngine::Precision netPrecision;
        InferenceEngine::SizeVector inputShape0, inputShape1;
        std::shared_ptr<ov::Node> binaryEltwise;
        size_t num_nodes, num_subgraphs;
        std::string targetDevice;
        std::tie(netPrecision, inputShape0, inputShape1, binaryEltwise,
                 num_nodes, num_subgraphs, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS[0]=" << CommonTestUtils::vec2str(inputShape0) << "_";
        result << "IS[1]=" << CommonTestUtils::vec2str(inputShape1) << "_";
        result << "Op=" << binaryEltwise->get_type_name() << "_";
        result << "#N=" << num_nodes << "_";
        result << "#S=" << num_subgraphs << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // the simplest possible eltwise operation with streaming access to the data
    void ConvEltwise::SetUp() {
        std::vector<size_t> inputShape0, inputShape1;
        InferenceEngine::Precision netPrecision;
        std::shared_ptr<ov::Node> binaryEltwise;
        std::tie(netPrecision, inputShape0, inputShape1, binaryEltwise,
                 ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();

        init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});
        std::vector<std::shared_ptr<ov::Node>> eltwiseOps {binaryEltwise,
                                                       std::make_shared<ov::op::v0::Abs>(),
                                                       std::make_shared<ov::op::v0::Sqrt>()};
        const auto f  = ov::test::snippets::ConvMulActivation({inputShape0, inputShape1}, eltwiseOps);
        function = f.getOriginal();
    }

TEST_P(ConvEltwise, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
};

}  // namespace LayerTestsDefinitions
