// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

class TileWithTwoOutputEdges : public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        auto ngPrc = element::f32;
        ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 3, 12, 9})};

        std::vector<int64_t> repeats = {1, 2, 1, 1};
        auto repeatsNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, std::vector<size_t>{repeats.size()}, repeats);
        auto tile = std::make_shared<ov::op::v0::Tile>(inputParams[0], repeatsNode);

        const auto const1 = ov::test::utils::make_constant(ngPrc, std::vector<size_t>{1, 6, 1, 1});
        const auto const2 = ov::test::utils::make_constant(ngPrc, std::vector<size_t>{1, 6, 1, 1});

        const auto add1 = utils::make_eltwise(tile->output(0), const1, utils::EltwiseTypes::ADD);
        const auto add2 = utils::make_eltwise(tile->output(0), const2, utils::EltwiseTypes::ADD);

        NodeVector results{add1, add2};
        function = std::make_shared<ov::Model>(results, inputParams, "TileWithTwoOutputEdges");
    }
};

TEST_F(TileWithTwoOutputEdges, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
