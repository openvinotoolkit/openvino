// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

/*This test runs the following subgraph:

                      param
                        |
                        |
                      Split
                     /  |  \
                    /   |   \
                  Add  Add  Add
                    \   |   /\
                     \  |  /  \
                      Concat  Result
                     /  |  \            
                    /   |   \
                  Add  Add   Result
                   |    |
                  Add  Add
                  /     |
               Result  Result

The main purpose of the test is to check the memory sharing between result and in_place edges.
*/

namespace ov {
namespace test {

class SplitConcatAddInPlace : virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        ov::test::InputShape input_shape{{}, {{1, 3, 3, 3}}};
        init_input_shapes({input_shape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(params.front(), split_axis_op, 3);

        auto add_const = std::make_shared<ov::op::v0::Constant>(precision, ov::Shape{1}, std::vector<float>({1.0f}));
        auto add_1 = utils::make_eltwise(split->output(0), add_const, utils::EltwiseTypes::ADD);
        auto result_add_1 = std::make_shared<ov::op::v0::Result>(add_1);
        auto add_2 = utils::make_eltwise(split->output(1), add_const, utils::EltwiseTypes::ADD);
        auto add_3 = utils::make_eltwise(split->output(2), add_const, utils::EltwiseTypes::ADD);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{add_1, add_2, add_3}, 1);
        auto result_concat = std::make_shared<ov::op::v0::Result>(concat);
        auto add_4 = utils::make_eltwise(concat, add_const, utils::EltwiseTypes::ADD);
        auto add_5 = utils::make_eltwise(concat, add_const, utils::EltwiseTypes::ADD);
        auto result_1 = std::make_shared<ov::op::v0::Result>(add_4);
        auto result_2 = std::make_shared<ov::op::v0::Result>(add_5);
        ov::ResultVector results = {result_1, result_2, result_add_1, result_concat};
        function = std::make_shared<ov::Model>(results, params, "Subgraph");
    }
};

TEST_F(SplitConcatAddInPlace, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov