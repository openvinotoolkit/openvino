// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/manager.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "base_test.hpp"

#include "openvino/op/ops.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ======================= ExtractorsManagerTest Unit tests =======================
class ExtractorsManagerTest : public ExtractorsManager,
                              public SubgraphsDumperBaseTest {
protected:
    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        test_map = {
            { "test_matcher", SubgraphExtractor::Ptr(new SubgraphExtractor) },
        };
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            test_parameter->set_friendly_name("test_parameter_0");
            std::shared_ptr<ov::op::v0::Abs> test_abs =
                std::make_shared<ov::op::v0::Abs>(test_parameter);
            std::shared_ptr<ov::op::v0::Result> test_res =
                std::make_shared<ov::op::v0::Result>(test_abs);
            test_model_0_0 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                                         ov::ParameterVector{test_parameter});
        }
    }

    ExtractorsManager::ExtractorsMap test_map;
    std::shared_ptr<ov::Model> test_model_0_0;
};

TEST_F(ExtractorsManagerTest, constructor) {
    OV_ASSERT_NO_THROW(auto m = ExtractorsManager());
    OV_ASSERT_NO_THROW(auto m = ExtractorsManager(test_map));
}

TEST_F(ExtractorsManagerTest, set_extractors) {
    OV_ASSERT_NO_THROW(this->set_extractors(test_map));
    ASSERT_EQ(this->m_extractors, test_map);
}

TEST_F(ExtractorsManagerTest, get_extractors) {
    OV_ASSERT_NO_THROW(this->set_extractors(test_map));
    OV_ASSERT_NO_THROW(this->get_extractors());
    ASSERT_EQ(this->m_extractors, this->get_extractors());
}

TEST_F(ExtractorsManagerTest, extract) {
    this->set_extractors(test_map);
    OV_ASSERT_NO_THROW(this->extract(test_model_0_0));
}

}  // namespace
