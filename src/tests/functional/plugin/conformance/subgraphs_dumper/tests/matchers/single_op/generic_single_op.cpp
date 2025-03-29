// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ops.hpp"
#include "matchers/single_op/single_op.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

class SingleOpMatcherTest : public SubgraphsDumperBaseTest {
protected:
    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        matcher = SingleOpMatcher();
    }

    SingleOpMatcher matcher;
};


// Check that different values of constant nodes on port 0 (default value) are ignored in match()
TEST_F(SingleOpMatcherTest, AllPortsAreConsts_IgnoreConstPortVals) {
    const auto const1 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({5, 5}), 1);
    const auto shape_pattern = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape({2}), std::vector<int>{1, 25});
    const auto op1 = std::make_shared<ov::op::v1::Reshape>(const1, shape_pattern, false);

    const auto const2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({5, 5}), 2);
    const auto op2 = std::make_shared<ov::op::v1::Reshape>(const2, shape_pattern, false);
    ASSERT_TRUE(matcher.match(op1, op2));
}

// Check match of equal nodes
TEST_F(SingleOpMatcherTest, AllPortsAreParams_NodesEqual) {
    const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10}));
    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 20}));
    const auto op1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param1, param2}), 1);
    const auto op2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param1, param2}), 1);
    ASSERT_TRUE(matcher.match(op1, op2));
}

// Check nodes doesn't match - different input ranks
TEST_F(SingleOpMatcherTest, AllPortsAreParams_RanksNotEqual) {
    const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10}));
    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 20}));
    const auto op1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param1, param2}), 1);

    const auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 40, 10}));
    const auto param4 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 40, 10}));
    const auto op2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param3, param4}), 1);
    ASSERT_FALSE(matcher.match(op1, op2));
}

// Check nodes doesn't match - different input element types
TEST_F(SingleOpMatcherTest, AllPortsAreParams_TypesNotEqual) {
    const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10}));
    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 20}));
    const auto op1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param1, param2}), 1);

    const auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f16, ov::Shape({10, 10}));
    const auto param4 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f16, ov::Shape({10, 20}));
    const auto op2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param3, param4}), 1);
    ASSERT_FALSE(matcher.match(op1, op2));
}

// Check nodes doesn't match - different input element types
TEST_F(SingleOpMatcherTest, AllPortsAreParams_AttrsNotEqual) {
    const auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10, 10}));
    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10, 10}));
    const auto op1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param1, param2}), 1);

    const auto param3 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10, 10}));
    const auto param4 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({10, 10, 10}));
    const auto op2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector({param3, param4}), 2);
    ASSERT_FALSE(matcher.match(op1, op2));
}

// Check nodes Add OPs match with different constants on ports
TEST_F(SingleOpMatcherTest, ChecAddOpConfiguration) {
    const auto const1 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({5, 5}), 1);
    const auto const2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({5, 5}), 2);
    const auto op1 = std::make_shared<ov::op::v1::Add>(const1, const2);

    const auto const3 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({5, 5}), 3);
    const auto const4 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({5, 5}), 4);
    const auto op2  = std::make_shared<ov::op::v1::Add>(const1, const2);
    ASSERT_TRUE(matcher.match(op1, op2));
}

} // namespace
