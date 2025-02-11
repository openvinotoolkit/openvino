// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/subgraph.hpp"
#include "utils/model_comparator.hpp"
#include "base_test.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/core/model.hpp"
#include "openvino/openvino.hpp"
#include "utils/model.hpp"
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
        ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();
        model_comparator->set_shape_strict_match(false);
        model_comparator->set_match_coefficient(0.9f);
    }

    std::shared_ptr<ov::Model> test_model_0_0, test_model_0_1, test_model_1;
};

TEST_F(ModelComparatorTest, get) {
    ov::util::ModelComparator::Ptr model_comparator = nullptr;
    OV_ASSERT_NO_THROW(model_comparator = ov::util::ModelComparator::get());
    ASSERT_EQ(model_comparator, ov::util::ModelComparator::get());
}

TEST_F(ModelComparatorTest, match) {
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_0_1));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_1));
    ASSERT_FALSE(model_comparator->match(test_model_0_0, test_model_1));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_1, test_model_1));
    ASSERT_FALSE(model_comparator->match(test_model_0_1, test_model_1));
}

TEST_F(ModelComparatorTest, match_strict_shape) {
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();
    OV_ASSERT_NO_THROW(model_comparator->set_shape_strict_match(true));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1));
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
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();
    model_comparator->set_match_coefficient(0.5f);
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_0_1));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_0, test_model_1));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_1, test_model_1));
    ASSERT_TRUE(model_comparator->match(test_model_0_1, test_model_1));
}

TEST_F(ModelComparatorTest, match_with_in_info) {
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();
    std::map<std::string, ov::conformance::InputInfo>
        test_in_info({{"test_parameter_0", ov::conformance::InputInfo(ov::Shape{1, 2})}}),
        test_in_info_({{"test_parameter_0", ov::conformance::InputInfo(ov::Shape{1, 2})}}),
        test_in_info_1({{"test_parameter_1", ov::conformance::InputInfo(ov::Shape{2, 5}, 1, 2, true)}});
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_));
    ASSERT_TRUE(std::get<0>(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_)));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_1));
    ASSERT_FALSE(std::get<0>(model_comparator->match(test_model_0_0, test_model_0_1, test_in_info, test_in_info_1)));
    OV_ASSERT_NO_THROW(model_comparator->match(test_model_0_1, test_model_1, test_in_info, test_in_info));
    ASSERT_FALSE(std::get<0>(model_comparator->match(test_model_0_1, test_model_1, test_in_info, test_in_info)));
}

TEST_F(ModelComparatorTest, is_subgraph) {
    ov::util::ModelComparator::Ptr model_comparator = ov::util::ModelComparator::get();
    OV_ASSERT_NO_THROW(model_comparator->is_subgraph(test_model_0_0, test_model_0_1));
    auto is_subgraph = model_comparator->is_subgraph(test_model_0_0, test_model_0_1);
    ASSERT_TRUE(std::get<0>(is_subgraph));
    OV_ASSERT_NO_THROW(model_comparator->is_subgraph(test_model_0_0, test_model_1));
    ASSERT_FALSE(std::get<0>(model_comparator->is_subgraph(test_model_0_0, test_model_1)));
    OV_ASSERT_NO_THROW(model_comparator->is_subgraph(test_model_0_1, test_model_1));
    ASSERT_FALSE(std::get<0>(model_comparator->is_subgraph(test_model_0_1, test_model_1)));
}

}  // namespace
