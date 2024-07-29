// // Copyright (C) 2018-2024 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include "matchers/subgraph/subgraph.hpp"
// #include "base_test.hpp"

// #include "openvino/op/abs.hpp"
// #include "openvino/op/relu.hpp"
// #include "openvino/op/parameter.hpp"
// #include "openvino/op/result.hpp"

// namespace {

// using namespace ov::tools::subgraph_dumper;

// // ======================= ExtractorsManagerTest Unit tests =======================
// class SubgraphExtractorTest : public SubgraphExtractor,
//                               public SubgraphsDumperBaseTest {
// protected:
//     void SetUp() override {
//         SubgraphsDumperBaseTest::SetUp();
//         {
//             std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//                 std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
//             std::shared_ptr<ov::op::v0::Abs> test_abs =
//                 std::make_shared<ov::op::v0::Abs>(test_parameter);
//             std::shared_ptr<ov::op::v0::Result> test_res =
//                 std::make_shared<ov::op::v0::Result>(test_abs);
//             test_model_0_0 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                          ov::ParameterVector{test_parameter});
//         }
//         {
//             std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//                 std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 5});
//             std::shared_ptr<ov::op::v0::Abs> test_abs =
//                 std::make_shared<ov::op::v0::Abs>(test_parameter);
//             std::shared_ptr<ov::op::v0::Result> test_res =
//                 std::make_shared<ov::op::v0::Result>(test_abs);
//             test_model_0_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                          ov::ParameterVector{test_parameter});
//         }
//         {
//             std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//                 std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 5});
//             std::shared_ptr<ov::op::v0::Relu> test_abs =
//                 std::make_shared<ov::op::v0::Relu>(test_parameter);
//             std::shared_ptr<ov::op::v0::Result> test_res =
//                 std::make_shared<ov::op::v0::Result>(test_abs);
//             test_model_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                        ov::ParameterVector{test_parameter});
//         }
//     }

//     std::shared_ptr<ov::Model> test_model_0_0, test_model_0_1, test_model_1;
// };

// TEST_F(SubgraphExtractorTest, match) {
//     OV_ASSERT_NO_THROW(this->match(test_model_0_0, test_model_0_1));
//     ASSERT_TRUE(this->match(test_model_0_0, test_model_0_1));
//     OV_ASSERT_NO_THROW(this->match(test_model_0_0, test_model_1));
//     ASSERT_FALSE(this->match(test_model_0_0, test_model_1));
//     OV_ASSERT_NO_THROW(this->match(test_model_0_1, test_model_1));
//     ASSERT_FALSE(this->match(test_model_0_1, test_model_1));
// }

// TEST_F(SubgraphExtractorTest, match_90_percent) {
//     {
//         std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//             std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
//         std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
//             std::make_shared<ov::op::v0::Abs>(test_parameter);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_0);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_2 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_1);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_3 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_2);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_4 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_3);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_5 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_4);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_6 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_5);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_7 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_6);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_8 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_7);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_9 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_8);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_10 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_9);
//         std::shared_ptr<ov::op::v0::Result> test_res =
//             std::make_shared<ov::op::v0::Result>(test_abs_10);
//         test_model_0_0 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                         ov::ParameterVector{test_parameter});
//     }
//     {
//         std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//             std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
//         std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
//             std::make_shared<ov::op::v0::Abs>(test_parameter);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_0);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_2 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_1);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_3 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_2);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_4 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_3);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_5 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_4);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_6 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_5);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_7 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_6);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_8 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_7);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_9 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_8);
//         std::shared_ptr<ov::op::v0::Relu> test_abs_10 =
//             std::make_shared<ov::op::v0::Relu>(test_abs_9);
//         std::shared_ptr<ov::op::v0::Result> test_res =
//             std::make_shared<ov::op::v0::Result>(test_abs_10);
//         test_model_0_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                         ov::ParameterVector{test_parameter});
//     }
//     {
//         std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//             std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
//         std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
//             std::make_shared<ov::op::v0::Abs>(test_parameter);
//         std::shared_ptr<ov::op::v0::Relu> test_abs_1 =
//             std::make_shared<ov::op::v0::Relu>(test_abs_0);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_2 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_1);
//         std::shared_ptr<ov::op::v0::Relu> test_abs_3 =
//             std::make_shared<ov::op::v0::Relu>(test_abs_2);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_4 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_3);
//         std::shared_ptr<ov::op::v0::Relu> test_abs_5 =
//             std::make_shared<ov::op::v0::Relu>(test_abs_4);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_6 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_5);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_7 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_6);
//         std::shared_ptr<ov::op::v0::Relu> test_abs_8 =
//             std::make_shared<ov::op::v0::Relu>(test_abs_7);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_9 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_8);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_10 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_9);
//         std::shared_ptr<ov::op::v0::Result> test_res =
//             std::make_shared<ov::op::v0::Result>(test_abs_10);
//         test_model_1 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                    ov::ParameterVector{test_parameter});
//     }
//     OV_ASSERT_NO_THROW(this->match(test_model_0_0, test_model_0_1));
//     ASSERT_TRUE(this->match(test_model_0_0, test_model_0_1));
//     OV_ASSERT_NO_THROW(this->match(test_model_0_0, test_model_1));
//     ASSERT_FALSE(this->match(test_model_0_0, test_model_1));
//     OV_ASSERT_NO_THROW(this->match(test_model_0_1, test_model_1));
//     ASSERT_FALSE(this->match(test_model_0_1, test_model_1));
// }

// TEST_F(SubgraphExtractorTest, extract) {
//     OV_ASSERT_NO_THROW(this->extract(test_model_0_0));
//     OV_ASSERT_NO_THROW(this->extract(test_model_0_1));
//     OV_ASSERT_NO_THROW(this->extract(test_model_1));
// }

// TEST_F(SubgraphExtractorTest, is_subgraph) {
//     auto is_subgraph = this->is_subgraph(test_model_0_0, test_model_0_0);
//     OV_ASSERT_NO_THROW(this->is_subgraph(test_model_0_0, test_model_0_0));
//     ASSERT_TRUE(std::get<0>(is_subgraph));
//     OV_ASSERT_NO_THROW(this->is_subgraph(test_model_0_0, test_model_1));
//     is_subgraph = this->is_subgraph(test_model_0_0, test_model_1);
//     ASSERT_FALSE(std::get<0>(is_subgraph));
//     OV_ASSERT_NO_THROW(this->is_subgraph(test_model_0_1, test_model_1));
//     is_subgraph = this->is_subgraph(test_model_0_1, test_model_1);
//     ASSERT_FALSE(std::get<0>(is_subgraph));
//     {
//         std::shared_ptr<ov::op::v0::Parameter> test_parameter =
//             std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
//         std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
//             std::make_shared<ov::op::v0::Abs>(test_parameter);
//         std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
//             std::make_shared<ov::op::v0::Abs>(test_abs_0);
//         std::shared_ptr<ov::op::v0::Result> test_res =
//             std::make_shared<ov::op::v0::Result>(test_abs_1);
//         auto big_model_0 = std::make_shared<ov::Model>(ov::ResultVector{test_res},
//                                                         ov::ParameterVector{test_parameter});
//         is_subgraph = this->is_subgraph(test_model_0_0, big_model_0);
//         OV_ASSERT_NO_THROW(this->is_subgraph(test_model_0_0, big_model_0));
//         ASSERT_TRUE(std::get<0>(is_subgraph));
//         ASSERT_EQ(std::get<1>(is_subgraph), big_model_0);
//         ASSERT_EQ(std::get<2>(is_subgraph), test_model_0_0);

//         is_subgraph = this->is_subgraph(test_model_0_1, big_model_0);
//         OV_ASSERT_NO_THROW(this->is_subgraph(test_model_0_1, big_model_0));
//         ASSERT_TRUE(std::get<0>(is_subgraph));
//         ASSERT_EQ(std::get<1>(is_subgraph), big_model_0);
//         ASSERT_EQ(std::get<2>(is_subgraph), test_model_0_1);
//         OV_ASSERT_NO_THROW(this->is_subgraph(test_model_1, big_model_0));
//         ASSERT_FALSE(std::get<0>(this->is_subgraph(test_model_1, big_model_0)));
//     }
// }

// }  // namespace
