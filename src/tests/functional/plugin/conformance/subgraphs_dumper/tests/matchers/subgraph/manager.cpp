// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/manager.hpp"
#include "matchers/subgraph/subgraph.hpp"
#include "base_test.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

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
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 5});
            test_parameter->set_friendly_name("test_parameter_1");
            std::shared_ptr<ov::op::v0::Abs> test_abs =
                std::make_shared<ov::op::v0::Abs>(test_parameter);
            std::shared_ptr<ov::op::v0::Result> test_res =
                std::make_shared<ov::op::v0::Result>(test_abs);
            test_model_0_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                                         ov::ParameterVector{test_parameter});
        }
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 5});
            std::shared_ptr<ov::op::v0::Relu> test_abs =
                std::make_shared<ov::op::v0::Relu>(test_parameter);
            std::shared_ptr<ov::op::v0::Result> test_res =
                std::make_shared<ov::op::v0::Result>(test_abs);
            test_model_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                                       ov::ParameterVector{test_parameter});
        }
    }

    ExtractorsManager::ExtractorsMap test_map;
    std::shared_ptr<ov::Model> test_model_0_0, test_model_0_1, test_model_1;
};

TEST_F(ExtractorsManagerTest, constructor) {
    ASSERT_NO_THROW(auto m = ExtractorsManager());
    ASSERT_NO_THROW(auto m = ExtractorsManager(test_map));
}

TEST_F(ExtractorsManagerTest, set_extractors) {
    ASSERT_NO_THROW(this->set_extractors(test_map));
    ASSERT_EQ(this->m_extractors, test_map);
}

TEST_F(ExtractorsManagerTest, get_extractors) {
    ASSERT_NO_THROW(this->set_extractors(test_map));
    ASSERT_NO_THROW(this->get_extractors());
    ASSERT_EQ(this->m_extractors, this->get_extractors());
}

TEST_F(ExtractorsManagerTest, match) {
    this->set_extractors(test_map);
    ASSERT_NO_THROW(this->match(test_model_0_0, test_model_0_1));
    ASSERT_TRUE(this->match(test_model_0_0, test_model_0_1));
    ASSERT_NO_THROW(this->match(test_model_0_0, test_model_1));
    ASSERT_FALSE(this->match(test_model_0_0, test_model_1));
    ASSERT_NO_THROW(this->match(test_model_0_1, test_model_1));
    ASSERT_FALSE(this->match(test_model_0_1, test_model_1));
}

TEST_F(ExtractorsManagerTest, is_subgraph) {
    this->set_extractors(test_map);
    ASSERT_NO_THROW(this->is_subgraph(test_model_0_0, test_model_0_1));
    auto is_subgraph = this->is_subgraph(test_model_0_0, test_model_0_1);
    ASSERT_TRUE(std::get<0>(is_subgraph));
    ASSERT_NO_THROW(this->is_subgraph(test_model_0_0, test_model_1));
    ASSERT_FALSE(std::get<0>(this->is_subgraph(test_model_0_0, test_model_1)));
    ASSERT_NO_THROW(this->is_subgraph(test_model_0_1, test_model_1));
    ASSERT_FALSE(std::get<0>(this->is_subgraph(test_model_0_1, test_model_1)));
}

TEST_F(ExtractorsManagerTest, match_with_in_info) {
    this->set_extractors(test_map);
    std::map<std::string, InputInfo> test_in_info({{"test_parameter_0", InputInfo()}}), test_in_info_1({{"test_parameter_1", InputInfo({}, 1, 2, true)}});
    ASSERT_NO_THROW(this->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info));
    ASSERT_TRUE(this->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info));
    ASSERT_NO_THROW(this->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_1));
    ASSERT_FALSE(this->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_1));
    ASSERT_NO_THROW(this->match(test_model_0_1, test_model_1, test_in_info, test_in_info));
    ASSERT_FALSE(this->match(test_model_0_1, test_model_1, test_in_info, test_in_info));
}

TEST_F(ExtractorsManagerTest, extract) {
    this->set_extractors(test_map);
    ASSERT_NO_THROW(this->extract(test_model_0_0));
}

TEST_F(ExtractorsManagerTest, align_input_info) {
    std::map<std::string, InputInfo> test_in_info({{"test_parameter_0", InputInfo()}}), test_in_info_ref({{"test_parameter_1", InputInfo()}});
    ASSERT_NE(test_in_info, test_in_info_ref);
    ASSERT_NO_THROW(this->align_input_info(test_model_0_0, test_model_0_1, test_in_info, test_in_info_ref));
    auto c = this->align_input_info(test_model_0_0, test_model_0_1, test_in_info, test_in_info_ref);
    ASSERT_EQ(c, test_in_info_ref);
}

TEST_F(ExtractorsManagerTest, align_input_info_for_subgraphs) {
    std::map<std::string, InputInfo> test_in_info({{"test_parameter_0", InputInfo()}}), test_in_info_ref({{"test_parameter_1", InputInfo()}});
    ASSERT_NE(test_in_info, test_in_info_ref);
    ASSERT_NO_THROW(this->align_input_info(test_model_0_0, test_model_0_1, test_in_info, test_in_info_ref, {{"test_parameter_0", "test_parameter_1"}}));
    auto c = this->align_input_info(test_model_0_0, test_model_0_1, test_in_info, test_in_info_ref);
    ASSERT_EQ(c, test_in_info_ref);
}

}  // namespace
