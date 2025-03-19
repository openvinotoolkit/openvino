// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"
#include "openvino/op/parameter.hpp"

#include "matchers/single_op/manager.hpp"
#include "matchers/single_op/single_op.hpp"

#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ======================= MatcherManager Unit tests =======================
class MatchersManagerTest : public MatchersManager,
                            public SubgraphsDumperBaseTest {
protected:
    MatchersManager::MatchersMap test_map;
    std::shared_ptr<ov::op::v0::Abs> test_abs;
    std::shared_ptr<ov::op::v0::Parameter> test_parameter;

    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        test_map = {
            { "test_matcher", SingleOpMatcher::Ptr(new SingleOpMatcher) },
        };
        test_parameter =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
        test_abs =
            std::make_shared<ov::op::v0::Abs>(test_parameter);
    }
};

TEST_F(MatchersManagerTest, constructor) {
    OV_ASSERT_NO_THROW(auto m = MatchersManager());
    OV_ASSERT_NO_THROW(auto m = MatchersManager(test_map));
}

TEST_F(MatchersManagerTest, set_matchers) {
    OV_ASSERT_NO_THROW(this->set_matchers(test_map));
    ASSERT_EQ(this->m_matchers, test_map);
}

TEST_F(MatchersManagerTest, get_matchers) {
    OV_ASSERT_NO_THROW(this->set_matchers(test_map));
    OV_ASSERT_NO_THROW(this->get_matchers());
    ASSERT_EQ(this->m_matchers, this->get_matchers());
}

TEST_F(MatchersManagerTest, get_config) {
    OV_ASSERT_NO_THROW(this->get_config(test_abs));
}

TEST_F(MatchersManagerTest, match) {
    this->set_matchers(test_map);
    OV_ASSERT_NO_THROW(this->match(test_parameter, test_abs));
    OV_ASSERT_NO_THROW(this->match(test_abs, test_abs));
    ASSERT_TRUE(this->match(test_abs, test_abs));
    ASSERT_TRUE(this->match(test_parameter, test_parameter));
    ASSERT_FALSE(this->match(test_parameter, test_abs));
}

}  // namespace
