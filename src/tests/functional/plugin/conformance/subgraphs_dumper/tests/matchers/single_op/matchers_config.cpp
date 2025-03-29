// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op/single_op.hpp"
#include "openvino/op/ops.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::op;
using ov::element::Type_t;
using namespace ov::tools::subgraph_dumper;

class MatcherConfigTest : public SubgraphsDumperBaseTest {
protected:
    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        const auto const1 = std::make_shared<v0::Constant>(Type_t::f32, ov::Shape({5, 5}), 1);
        const auto const2 = std::make_shared<v0::Constant>(Type_t::f32, ov::Shape({5, 5}), 2);
        node = std::make_shared<v1::Add>(const1, const2);
    }

    std::shared_ptr<ov::Node> node;
};


// Check that matcher configuration for operation created successfully and all parameters are set
TEST_F(MatcherConfigTest, ParametersAreSet) {
    std::vector<size_t> ignored_ports = {0};
    std::vector<std::string> ignored_attrs = {"attr"};
    MatcherConfig<v1::Add> matcher_cfg(ignored_attrs, ignored_ports);
    ASSERT_TRUE(matcher_cfg.op_in_config(node));
    ASSERT_TRUE(matcher_cfg.ignored_ports == ignored_ports);
    ASSERT_TRUE(matcher_cfg.ignored_attributes == ignored_attrs);
    ASSERT_FALSE(matcher_cfg.is_fallback_config);
}

// Check that fallback matcher configuration created successfully and all parameters are set
TEST_F(MatcherConfigTest, FallbackConfig) {
    std::vector<size_t> ignored_ports = {0};
    std::vector<std::string> ignored_attrs = {"attr"};
    MatcherConfig<> matcher_cfg(ignored_attrs, ignored_ports);
    ASSERT_FALSE(matcher_cfg.op_in_config(node));
    ASSERT_TRUE(matcher_cfg.ignored_ports == ignored_ports);
    ASSERT_TRUE(matcher_cfg.ignored_attributes == ignored_attrs);
    ASSERT_TRUE(matcher_cfg.is_fallback_config);
}

// Check that fallback matcher configuration created with default constructor
TEST_F(MatcherConfigTest, FallbackConfigDefaultConstructor) {
    std::vector<size_t> ignored_ports = {};
    std::vector<std::string> ignored_attrs = {};
    auto matcher_cfg = MatcherConfig<>();
    ASSERT_FALSE(matcher_cfg.op_in_config(node));
    ASSERT_TRUE(matcher_cfg.ignored_ports == ignored_ports);
    ASSERT_TRUE(matcher_cfg.ignored_attributes == ignored_attrs);
    ASSERT_TRUE(matcher_cfg.is_fallback_config);
}

} // namespace
