// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/subgraph/subgraph.hpp"
#include "base_test.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

// ======================= ExtractorsManagerTest Unit tests =======================
class SubgraphExtractorTest : public SubgraphExtractor,
                              public SubgraphsDumperBaseTest {
protected:
    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
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

    std::shared_ptr<ov::Model> test_model_0_0, test_model_0_1, test_model_1;
};

TEST_F(SubgraphExtractorTest, match) {
    ASSERT_NO_THROW(this->match(test_model_0_0, test_model_0_1));
    ASSERT_TRUE(this->match(test_model_0_0, test_model_0_1));
    ASSERT_NO_THROW(this->match(test_model_0_0, test_model_1));
    ASSERT_FALSE(this->match(test_model_0_0, test_model_1));
    ASSERT_NO_THROW(this->match(test_model_0_1, test_model_1));
    ASSERT_FALSE(this->match(test_model_0_1, test_model_1));
}

TEST_F(SubgraphExtractorTest, extract) {
    ASSERT_NO_THROW(this->extract(test_model_0_0));
    ASSERT_NO_THROW(this->extract(test_model_0_1));
    ASSERT_NO_THROW(this->extract(test_model_1));
}

}  // namespace
