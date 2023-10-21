// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/subgraph.hpp"
#include "utils/model_comparator.hpp"
#include "base_test.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ======================= ExtractorsManagerTest Unit tests =======================
class ModelComparatorTest : public SubgraphsDumperBaseTest {
protected:
    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
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

    void TearDown() override {
        ModelComparator::Ptr model_comparator = ModelComparator::get();
        model_comparator->set_shape_strict_match(false);
        model_comparator->set_match_coefficient(0.9f);
    }

    std::shared_ptr<ov::Model> test_model_0_0, test_model_0_1, test_model_1;
};

TEST_F(ModelComparatorTest, get) {
    ModelComparator::Ptr model_comparator = nullptr;
    ASSERT_NO_THROW(model_comparator = ModelComparator::get());
    ASSERT_EQ(model_comparator, ModelComparator::get());
}

TEST_F(ModelComparatorTest, match) {
    ModelComparator::Ptr model_comparator = ModelComparator::get();
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_1));
    ASSERT_FALSE(model_comparator->match(test_model_0_0, test_model_1));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_1, test_model_1));
    ASSERT_FALSE(model_comparator->match(test_model_0_1, test_model_1));
}

TEST_F(ModelComparatorTest, match_strict_shape) {
    ModelComparator::Ptr model_comparator = ModelComparator::get();
    ASSERT_NO_THROW(model_comparator->set_shape_strict_match(true));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_FALSE(model_comparator->match(test_model_0_0, test_model_0_1));
    {
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            test_parameter->set_friendly_name("test_parameter_1");
            std::shared_ptr<ov::op::v0::Abs> test_abs =
                std::make_shared<ov::op::v0::Abs>(test_parameter);
            std::shared_ptr<ov::op::v0::Result> test_res =
                std::make_shared<ov::op::v0::Result>(test_abs);
            test_model_0_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                                         ov::ParameterVector{test_parameter});
        }
        ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_0_1));
    }
}

TEST_F(ModelComparatorTest, match_with_low_coeff) {
    ModelComparator::Ptr model_comparator = ModelComparator::get();
    model_comparator->set_match_coefficient(0.5f);
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_1));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_1, test_model_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_1, test_model_1));
}

TEST_F(ModelComparatorTest, match_with_in_info) {
    ModelComparator::Ptr model_comparator = ModelComparator::get();
    std::map<std::string, InputInfo> test_in_info({{"test_parameter_0", InputInfo()}}),
                                     test_in_info_1({{"test_parameter_1", InputInfo({}, 1, 2, true)}});
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info));
    ASSERT_TRUE(std::get<0>(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info)));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_1));
    ASSERT_FALSE(std::get<0>(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_1)));
    ASSERT_NO_THROW(model_comparator->match(test_model_0_1, test_model_1, test_in_info, test_in_info));
    ASSERT_FALSE(std::get<0>(model_comparator->match(test_model_0_1, test_model_1, test_in_info, test_in_info)));
}

TEST_F(ModelComparatorTest, is_subgraph) {
    ModelComparator::Ptr model_comparator = ModelComparator::get();
    ASSERT_NO_THROW(model_comparator->is_subgraph(test_model_0_0, test_model_0_1));
    auto is_subgraph = model_comparator->is_subgraph(test_model_0_0, test_model_0_1);
    ASSERT_TRUE(std::get<0>(is_subgraph));
    ASSERT_NO_THROW(model_comparator->is_subgraph(test_model_0_0, test_model_1));
    ASSERT_FALSE(std::get<0>(model_comparator->is_subgraph(test_model_0_0, test_model_1)));
    ASSERT_NO_THROW(model_comparator->is_subgraph(test_model_0_1, test_model_1));
    ASSERT_FALSE(std::get<0>(model_comparator->is_subgraph(test_model_0_1, test_model_1)));
}

}  // namespace
