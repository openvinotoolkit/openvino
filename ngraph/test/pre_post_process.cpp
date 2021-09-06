// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/pre_post_process/pre_post_process.hpp"

#include <ngraph/pass/visualize_tree.hpp>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace ngraph::test;

static std::shared_ptr<Function> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<ngraph::op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    auto res = std::make_shared<ngraph::op::v0::Result>(data1);
    res->set_friendly_name("Result");
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{res}, ngraph::ParameterVector{data1});
}

TEST(pre_post_process, simple_mean_scale) {
    auto f = create_simple_function(ngraph::element::f32, ngraph::Shape{1, 3, 2, 2});
    f = PrePostProcessor().in(InputInfo().preprocess(PreProcessSteps().mean(1.f).scale(2.f))).build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result},
                {make_host_tensor<ngraph::element::f32>(ngraph::Shape{1, 3, 2, 2},
                                                        {1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.}, result_val));
}

TEST(pre_post_process, scale_then_mean) {
    auto f = create_simple_function(ngraph::element::f32, ngraph::Shape{1, 3, 2, 2});
    f = PrePostProcessor().in(InputInfo().preprocess(PreProcessSteps().scale(2.0f).mean(1.0f))).build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result},
                {make_host_tensor<ngraph::element::f32>(ngraph::Shape{1, 3, 2, 2},
                                                        {2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 100., 200.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 49., 99.}, result_val));
}

TEST(pre_post_process, convert_element_type_and_scale) {
    auto f = create_simple_function(element::i8, ngraph::Shape{1, 3, 2, 2});
    f = PrePostProcessor()
            .in(InputInfo()
                    .tensor(InputTensorInfo().set_element_type(element::i16))
                    .preprocess(PreProcessSteps()
                                    .convert_element_type(element::f32)
                                    .scale(2.f)
                                    .convert_element_type(element::i8)))
            .build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result},
                {make_host_tensor<ngraph::element::i32>(ngraph::Shape{1, 3, 2, 2},
                                                        {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 1000000, 200})});
    auto result_val = read_vector<int8_t>(result);
    EXPECT_TRUE(all_close(std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (int8_t)500000, 100}, result_val));
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i16);

    ASSERT_EQ(f->get_output_element_type(0), element::i8);
}

// TEST(pre_post_process, convert_element_type_from_unknown) {
//    auto f = create_simple_function(element::i32, ngraph::Shape{1, 3, 224, 224});
//    ASSERT_ANY_THROW(f = PrePostProcessor()
//                             .in(InputInfo()
//                                 .preprocess(PreProcessSteps()
//                                    .convert_element_type(element::i8)
//                                    .convert_element_type(element::i32)))
//                             .build(f););
//}

TEST(pre_post_process, convert_element_type_no_match) {
    auto f = create_simple_function(element::i32, ngraph::Shape{1, 3, 224, 224});
    ASSERT_ANY_THROW(f = PrePostProcessor()
                             .in(InputInfo()
                                     .tensor(InputTensorInfo().set_element_type(element::i32))
                                     .preprocess(PreProcessSteps().convert_element_type(element::f32).scale(2.0f)))
                             .build(f););
}

TEST(pre_post_process, tensor_element_type_and_scale) {
    auto f = create_simple_function(element::i8, ngraph::Shape{1, 3, 1, 1});
    f = PrePostProcessor()
            .in(InputInfo()
                    .tensor(InputTensorInfo().set_element_type(element::f32))
                    .preprocess(PreProcessSteps().scale(2.0f).convert_element_type(element::i8)))
            .build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result}, {make_host_tensor<ngraph::element::f32>(ngraph::Shape{1, 3, 1, 1}, {2., 4., 6.})});
    auto result_val = read_vector<int8_t>(result);
    EXPECT_TRUE(all_close(std::vector<int8_t>{1, 2, 3}, result_val));
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);

    ASSERT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, mean_scale_vector_network_layout) {
    auto f = create_simple_function(element::f32, ngraph::PartialShape{Dimension::dynamic(), 3, 2, 1});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    f = PrePostProcessor()
            .in(InputInfo()
                    .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f}))
                    .network(InputNetworkInfo().set_layout(PartialLayout("NCHW"))))
            .build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result},
                {make_host_tensor<ngraph::element::f32>(ngraph::Shape{1, 3, 2, 1}, {5., 1., 5., 11., 11., -1.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{2., 0., 1., 3., 2., -1.}, result_val));
}

TEST(pre_post_process, scale_vector_tensor_layout) {
    auto f = create_simple_function(element::f32, ngraph::PartialShape{Dimension::dynamic(), 3, 1, 3});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    f = PrePostProcessor()
            .in(InputInfo()
                    .tensor(InputTensorInfo().set_layout(PartialLayout("NHWC")))
                    .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f})))
            .build(f);
    auto result = std::make_shared<HostTensor>();
    f->evaluate(
        {result},
        {make_host_tensor<ngraph::element::f32>(ngraph::Shape{1, 3, 1, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{0., 0., 0., 3., 3., 3., 6., 6., 6.}, result_val));
}

TEST(pre_post_process, scale_vector_no_channels_layout) {
    auto f = create_simple_function(element::f32, ngraph::PartialShape{Dimension::dynamic(), 4, 4, 3});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_ANY_THROW(f = PrePostProcessor()
                             .in(InputInfo()
                                     .tensor(InputTensorInfo().set_layout(PartialLayout("NHW?")))
                                     .preprocess(PreProcessSteps().scale({0.1f, 0.2f, 0.3f})))
                             .build(f););
}
