// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/sqrt.hpp"
#include "snippets/conv_eltwise.hpp"
#include "subgraph_customizable.hpp"

namespace ov {
namespace test {
namespace snippets {

    std::string ConvEltwise::getTestCaseName(testing::TestParamInfo<ov::test::snippets::ConvEltwiseParams> obj) {
        ov::Shape inputShape0, inputShape1;
        std::shared_ptr<ov::Node> binaryEltwise;
        size_t num_nodes, num_subgraphs;
        std::string targetDevice;
        std::tie(inputShape0, inputShape1, binaryEltwise,
                 num_nodes, num_subgraphs, targetDevice) = obj.param;
        std::ostringstream result;
        result << "IS[0]=" << ov::test::utils::vec2str(inputShape0) << "_";
        result << "IS[1]=" << ov::test::utils::vec2str(inputShape1) << "_";
        result << "Op=" << binaryEltwise->get_type_name() << "_";
        result << "#N=" << num_nodes << "_";
        result << "#S=" << num_subgraphs << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // the simplest possible eltwise operation with streaming access to the data
    void ConvEltwise::SetUp() {
        ov::Shape inputShape0, inputShape1;
        std::shared_ptr<ov::Node> binaryEltwise;
        std::tie(inputShape0, inputShape1, binaryEltwise,
                 ref_num_nodes, ref_num_subgraphs, targetDevice) = this->GetParam();

        init_input_shapes({{{}, {inputShape0, }}, {{}, {inputShape1, }}});
        std::vector<std::shared_ptr<ov::Node>> eltwiseOps {binaryEltwise,
                                                       std::make_shared<ov::op::v0::Abs>(),
                                                       std::make_shared<ov::op::v0::Sqrt>()};
        const auto f  = ov::test::snippets::ConvMulActivationFunction({inputShape0, inputShape1}, eltwiseOps);
        function = f.getOriginal();
    }

TEST_P(ConvEltwise, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
};


} // namespace snippets
} // namespace test
} // namespace ov
