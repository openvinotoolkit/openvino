// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "matchers/single_op.hpp"
#include "ngraph/ops.hpp"
#include "functional_test_utils/include/functional_test_utils/layer_test_utils/op_info.hpp"

using namespace ngraph::op;
using namespace ngraph;
using ngraph::element::Type_t;

class SingleOpMatcherTest : public ::testing::Test {
protected:
    void SetUp() {
        matcher = SubgraphsDumper::SingleOpMatcher();
        op_info = LayerTestsUtils::OPInfo();
    }

    SubgraphsDumper::SingleOpMatcher matcher;
    LayerTestsUtils::OPInfo op_info;
};


// Check that different values of constant nodes on port 0 (default value) are ignored in match()
TEST_F(SingleOpMatcherTest, AllPortsAreConsts_IgnoreConstPortVals) {
    const auto const1 = std::make_shared<Constant>(Type_t::f32, Shape({5, 5}), 1);
    const auto shape_pattern = std::make_shared<Constant>(Type_t::i64, Shape({2}), std::vector<int>{1, 25});
    const auto op1 = std::make_shared<v1::Reshape>(const1, shape_pattern, false);

    const auto const2 = std::make_shared<Constant>(Type_t::f32, Shape({5, 5}), 2);
    const auto op2 = std::make_shared<v1::Reshape>(const2, shape_pattern, false);
    ASSERT_TRUE(matcher.match(op1, op2, op_info));
}

// Check match of equal nodes
TEST_F(SingleOpMatcherTest, AllPortsAreParams_NodesEqual) {
    const auto param1 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10}));
    const auto param2 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 20}));
    const auto op1 = std::make_shared<::Concat>(OutputVector({param1, param2}), 1);
    const auto op2 = std::make_shared<::Concat>(OutputVector({param1, param2}), 1);
    ASSERT_TRUE(matcher.match(op1, op2, op_info));
}

// Check nodes doesn't match - different input ranks
TEST_F(SingleOpMatcherTest, AllPortsAreParams_RanksNotEqual) {
    const auto param1 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10}));
    const auto param2 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 20}));
    const auto op1 = std::make_shared<::Concat>(OutputVector({param1, param2}), 1);

    const auto param3 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 40, 10}));
    const auto param4 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 40, 10}));
    const auto op2 = std::make_shared<::Concat>(OutputVector({param3, param4}), 1);
    ASSERT_FALSE(matcher.match(op1, op2, op_info));
}

// Check nodes doesn't match - different input element types
TEST_F(SingleOpMatcherTest, AllPortsAreParams_TypesNotEqual) {
    const auto param1 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10}));
    const auto param2 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 20}));
    const auto op1 = std::make_shared<::Concat>(OutputVector({param1, param2}), 1);

    const auto param3 = std::make_shared<Parameter>(element::Type_t::f16, Shape({10, 10}));
    const auto param4 = std::make_shared<Parameter>(element::Type_t::f16, Shape({10, 20}));
    const auto op2 = std::make_shared<::Concat>(OutputVector({param3, param4}), 1);
    ASSERT_FALSE(matcher.match(op1, op2, op_info));
}

// Check nodes doesn't match - different input element types
TEST_F(SingleOpMatcherTest, AllPortsAreParams_AttrsNotEqual) {
    const auto param1 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10, 10}));
    const auto param2 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10, 10}));
    const auto op1 = std::make_shared<::Concat>(OutputVector({param1, param2}), 1);

    const auto param3 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10, 10}));
    const auto param4 = std::make_shared<Parameter>(element::Type_t::f32, Shape({10, 10, 10}));
    const auto op2 = std::make_shared<::Concat>(OutputVector({param3, param4}), 2);
    ASSERT_FALSE(matcher.match(op1, op2, op_info));
}

// Check nodes Add OPs match with different constants on ports
TEST_F(SingleOpMatcherTest, ChecAddOpConfiguration) {
    const auto const1 = std::make_shared<Constant>(Type_t::f32, Shape({5, 5}), 1);
    const auto const2 = std::make_shared<Constant>(Type_t::f32, Shape({5, 5}), 2);
    const auto op1 = std::make_shared<v1::Add>(const1, const2);

    const auto const3 = std::make_shared<Constant>(Type_t::f32, Shape({5, 5}), 3);
    const auto const4 = std::make_shared<Constant>(Type_t::f32, Shape({5, 5}), 4);
    const auto op2  = std::make_shared<v1::Add>(const1, const2);
    ASSERT_TRUE(matcher.match(op1, op2, op_info));
}