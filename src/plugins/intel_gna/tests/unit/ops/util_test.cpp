// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>
#include <thread>
#include <memory>
#include <gtest/gtest.h>

#include "openvino/opsets/opset9.hpp"

#include "ops/util/util.hpp"

using namespace ov::intel_gna::ngraph_util;
using namespace ov::opset9;

namespace test {

using GnaOpsUtilTestParams = std::tuple<std::shared_ptr<ov::Node>, bool>;

class GnaOpsUtilTest : public ::testing::TestWithParam<GnaOpsUtilTestParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<GnaOpsUtilTestParams>& obj) {
        std::shared_ptr<ov::Node> node;  // node to be converted
        bool result; // expected result
        std::tie(node, result) = obj.param;

        std::ostringstream test_name;
        test_name << "Node=" << node->get_type_info().name << "_";
        test_name << "Op=" << node->get_type_info().get_version() << "_";
        test_name << "ExpectedResult=" << result;

        return test_name.str();
    }
};

class GnaOpsUtilIsPoolingTest : public GnaOpsUtilTest {
public:
    void validate() {
        std::shared_ptr<ov::Node> node;  // node to be converted
        bool result; // expected result
        std::tie(node, result) = GetParam();
        ASSERT_TRUE(result == is_pooling(node));
    }
};

class GnaOpsUtilIsEltwiseMulTest : public GnaOpsUtilTest {
public:
    void validate() {
        std::shared_ptr<ov::Node> node;  // node to be converted
        bool result; // expected result
        std::tie(node, result) = GetParam();
        ASSERT_TRUE(result == is_eltwise_mul(node));
    }
};

class GnaOpsUtilIsEltwiseAddTest : public GnaOpsUtilTest {
public:
    void validate() {
        std::shared_ptr<ov::Node> node;  // node to be converted
        bool result; // expected result
        std::tie(node, result) = GetParam();
        ASSERT_TRUE(result == is_eltwise_add(node));
    }
};

TEST_P(GnaOpsUtilIsPoolingTest, isPoolingTest) {
    validate();
}

TEST_P(GnaOpsUtilIsEltwiseMulTest, isEltwiseMulTest) {
    validate();
}

TEST_P(GnaOpsUtilIsEltwiseAddTest, isEltwiseAddTest) {
    validate();
}

ov::NodeVector pooling_nodes_false = {
    std::make_shared<VariadicSplit>(),
    std::make_shared<Concat>(),
    std::make_shared<MatMul>(),
    std::make_shared<ngraph::opset9::MaxPool>()
};

ov::NodeVector pooling_nodes_true = {
    std::make_shared<ngraph::opset7::MaxPool>()
};

ov::NodeVector eltwise_mul_nodes_false = {
    std::make_shared<VariadicSplit>(),
    std::make_shared<ngraph::op::Eltwise>(std::make_shared<Constant>(),
                                          std::make_shared<Constant>(),
                                          ELTWISE_TYPE::Sum),
};

ov::NodeVector eltwise_mul_nodes_true = {
    std::make_shared<ngraph::op::Eltwise>(std::make_shared<Constant>(),
                                          std::make_shared<Constant>(),
                                          ELTWISE_TYPE::Prod)
};

ov::NodeVector eltwise_add_nodes_false = {
    std::make_shared<VariadicSplit>(),
    std::make_shared<ngraph::op::Eltwise>(std::make_shared<Constant>(),
                                          std::make_shared<Constant>(),
                                          ELTWISE_TYPE::Prod)
};

ov::NodeVector eltwise_add_nodes_true = {
        std::make_shared<ngraph::op::Eltwise>(std::make_shared<Constant>(),
                                              std::make_shared<Constant>(),
                                              ELTWISE_TYPE::Sum)
};

INSTANTIATE_TEST_SUITE_P(smoke_ops_util_is_pooling,
                         GnaOpsUtilIsPoolingTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(pooling_nodes_true),
                            ::testing::Values(true)),
                         GnaOpsUtilIsPoolingTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(smoke_ops_util_not_pooling,
                         GnaOpsUtilIsPoolingTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(pooling_nodes_false),
                            ::testing::Values(false)),
                         GnaOpsUtilIsPoolingTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(smoke_ops_util_is_eltwise_mul,
                         GnaOpsUtilIsEltwiseMulTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(eltwise_mul_nodes_true),
                            ::testing::Values(true)),
                         GnaOpsUtilIsEltwiseMulTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(smoke_ops_util_not_elwise_mul,
                         GnaOpsUtilIsEltwiseMulTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(eltwise_mul_nodes_false),
                            ::testing::Values(false)),
                         GnaOpsUtilIsEltwiseMulTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(smoke_ops_util_is_eltwise_add,
                         GnaOpsUtilIsEltwiseAddTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(eltwise_add_nodes_true),
                            ::testing::Values(true)),
                         GnaOpsUtilIsEltwiseAddTest::get_test_name);

INSTANTIATE_TEST_SUITE_P(smoke_ops_util_not_eltwise_add,
                         GnaOpsUtilIsEltwiseAddTest,
                         ::testing::Combine(
                            ::testing::ValuesIn(eltwise_add_nodes_false),
                            ::testing::Values(false)),
                         GnaOpsUtilIsEltwiseAddTest::get_test_name);

} //namespace test