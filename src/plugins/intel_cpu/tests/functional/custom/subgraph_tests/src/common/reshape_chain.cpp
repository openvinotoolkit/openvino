// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class ReshapeChain : public SubgraphBaseTest {
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        InputShape inputShapes{{-1, -1, -1, -1}, {{10, 20, 30, 40}, {16, 24, 16, 24}, {4, 8, 12, 16}}};

        init_input_shapes({inputShapes});
        auto ngPrc = ov::element::f32;
        const auto secondInPrc = ov::element::Type_t::i32;
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
        }
        auto reshapeParam1 = std::make_shared<ov::op::v0::Constant>(secondInPrc, ov::Shape{3}, std::vector<int>{0, 0, -1});
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(inputParams.front(), reshapeParam1, true);
        auto reshapeParam2 = std::make_shared<ov::op::v0::Constant>(secondInPrc, ov::Shape{2}, std::vector<int>{0, -1});
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(reshape1, reshapeParam2, true);
        auto reshapeParam3 = std::make_shared<ov::op::v0::Constant>(secondInPrc, ov::Shape{1}, std::vector<int>{-1});
        auto reshape3 = std::make_shared<ov::op::v1::Reshape>(reshape2, reshapeParam3, true);
        auto reshapeParam4 = std::make_shared<ov::op::v0::Constant>(secondInPrc, ov::Shape{2}, std::vector<int>{4, -1});
        auto reshape4 = std::make_shared<ov::op::v1::Reshape>(reshape3, reshapeParam4, true);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape4)};
        function = std::make_shared<ov::Model>(results, inputParams, "reshapeChain");
    }
};

TEST_F(ReshapeChain, smoke_ReshapeChain) {
    run();
}

}  // namespace test
}  // namespace ov
