// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace ov;
using namespace ov::preprocess;
using namespace ngraph::test;

static std::shared_ptr<Function> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto res = std::make_shared<op::v0::Result>(data1);
    res->set_friendly_name("Result");
    return std::make_shared<Function>(ResultVector{res}, ParameterVector{data1});
}

static std::shared_ptr<Function> create_2inputs(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto data2 = std::make_shared<op::v0::Parameter>(type, shape);
    data2->set_friendly_name("input2");
    data1->get_output_tensor(0).set_names({"tensor_input2"});
    auto res1 = std::make_shared<op::v0::Result>(data1);
    res1->set_friendly_name("Result1");
    auto res2 = std::make_shared<op::v0::Result>(data2);
    res2->set_friendly_name("Result2");
    return std::make_shared<Function>(ResultVector{res1, res2}, ParameterVector{data1, data2});
}

TEST(pre_post_process, simple_mean_scale) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().mean(1.f).scale(2.f))).build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate(
        {result},
        {make_host_tensor<element::f32>(Shape{1, 3, 2, 2}, {1., 3., 5., 7., 9., 11., 13., 15., 17., 19., 21., 23.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.}, result_val));
}

TEST(pre_post_process, scale_then_mean) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().scale(2.0f).mean(2.0f))).build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result},
                {make_host_tensor<element::f32>(Shape{1, 3, 2, 2},
                                                {2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 100., 200.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{-1., 0, 1., 2., 3., 4., 5., 6., 7., 8., 48., 98.}, result_val));
}

TEST(pre_post_process, convert_element_type_and_scale) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 2, 2});
    f = PrePostProcessor()
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_element_type(element::i16))
                       .preprocess(PreProcessSteps()
                                       .convert_element_type(element::f32)
                                       .scale(2.f)
                                       .convert_element_type(element::i8)))
            .build(f);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i16);
    EXPECT_FALSE(f->get_parameters().front()->has_layout());
    EXPECT_EQ(f->get_output_element_type(0), element::i8);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result},
                {make_host_tensor<element::i16>(Shape{1, 3, 2, 2}, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 10000, 200})});
    auto result_val = read_vector<int8_t>(result);
    EXPECT_TRUE(all_close(std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, (int8_t)5000, 100}, result_val));
}

TEST(pre_post_process, convert_element_type_from_unknown) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    ASSERT_THROW(
        f = PrePostProcessor()
                .input(InputInfo().preprocess(
                    PreProcessSteps().convert_element_type(element::dynamic).convert_element_type(element::i32)))
                .build(f),
        ov::AssertFailure);
}

TEST(pre_post_process, convert_element_type_no_match) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    ASSERT_THROW(f = PrePostProcessor()
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_element_type(element::i32))
                                    .preprocess(PreProcessSteps().convert_element_type(element::f32).scale(2.0f)))
                         .build(f),
                 ov::AssertFailure);
}

TEST(pre_post_process, scale_not_float) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    ASSERT_THROW(
        f = PrePostProcessor()
                .input(InputInfo().preprocess(PreProcessSteps().convert_element_type(element::f32).scale(2.0f)))
                .build(f),
        ov::AssertFailure);
}

TEST(pre_post_process, mean_not_float) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    ASSERT_THROW(f = PrePostProcessor()
                         .input(InputInfo().preprocess(PreProcessSteps().convert_element_type(element::f32).mean(2.0f)))
                         .build(f),
                 ov::AssertFailure);
}

TEST(pre_post_process, tensor_element_type_and_scale) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
    f = PrePostProcessor()
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_element_type(element::f32))
                       .preprocess(PreProcessSteps().scale(2.0f).convert_element_type(element::i8)))
            .build(f);

    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
    EXPECT_FALSE(f->get_parameters().front()->has_layout());
    EXPECT_THROW(f->get_parameters().front()->get_layout(), ov::AssertFailure);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result}, {make_host_tensor<element::f32>(Shape{1, 3, 1, 1}, {2., 4., 6.})});
    auto result_val = read_vector<int8_t>(result);
    EXPECT_TRUE(all_close(std::vector<int8_t>{1, 2, 3}, result_val));
}

TEST(pre_post_process, custom_preprocessing) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 1, 1});
    f = PrePostProcessor()
            .input(InputInfo().preprocess(PreProcessSteps().custom([](const std::shared_ptr<Node>& node) {
                auto abs = std::make_shared<op::v0::Abs>(node);
                abs->set_friendly_name(node->get_friendly_name() + "/abs");
                return abs;
            })))
            .build(f);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result}, {make_host_tensor<element::i32>(Shape{1, 3, 1, 1}, {0, 4, -6})});
    auto result_val = read_vector<int32_t>(result);
    EXPECT_TRUE(all_close(std::vector<int32_t>{0, 4, 6}, result_val));
}

TEST(pre_post_process, test_lvalue) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    auto p = PrePostProcessor();
    auto p1 = std::move(p);
    p = std::move(p1);
    auto inputInfo = InputInfo();
    auto inputInfo2 = std::move(inputInfo);
    inputInfo = std::move(inputInfo2);
    {
        auto inputTensorInfo = InputTensorInfo();
        auto inputTensorInfo2 = std::move(inputTensorInfo);
        inputTensorInfo = std::move(inputTensorInfo2);
        auto& same = inputTensorInfo.set_element_type(element::f32);
        same.set_layout("?CHW");
        inputInfo.tensor(std::move(same));
    }
    {
        auto preprocessSteps = PreProcessSteps();
        auto preprocessSteps2 = std::move(preprocessSteps);
        preprocessSteps = std::move(preprocessSteps2);
        preprocessSteps.mean(1.f);
        preprocessSteps.scale(2.f);
        preprocessSteps.mean({1.f, 2.f, 3.f});
        preprocessSteps.scale({2.f, 3.f, 4.f});
        preprocessSteps.custom([](const std::shared_ptr<Node>& node) {
            auto abs = std::make_shared<op::v0::Abs>(node);
            abs->set_friendly_name(node->get_friendly_name() + "/abs");
            return abs;
        });
        auto& same = preprocessSteps.convert_element_type(element::i8);
        inputInfo.preprocess(std::move(same));
    }
    p.input(std::move(inputInfo));
    f = p.build(f);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_TRUE(f->get_parameters().front()->has_layout());
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "?CHW");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result}, {make_host_tensor<element::f32>(Shape{1, 3, 1, 1}, {-9., 17., -1.})});
    auto result_val = read_vector<int8_t>(result);
    EXPECT_TRUE(all_close(std::vector<int8_t>{3, 2, 1}, result_val));
}

TEST(pre_post_process, test_2_inputs_basic) {
    auto f = create_2inputs(element::f32, Shape{1, 3, 1, 1});
    { f = PrePostProcessor().input(InputInfo(1).preprocess(PreProcessSteps().mean(1.f).scale(2.0f))).build(f); }
    auto result1 = std::make_shared<HostTensor>();
    auto result2 = std::make_shared<HostTensor>();
    auto input1 = make_host_tensor<element::f32>(Shape{1, 3, 1, 1}, {3., 5., 7.});
    auto input2 = make_host_tensor<element::f32>(Shape{1, 3, 1, 1}, {3., 5., 7.});
    f->evaluate({result1, result2}, {input1, input2});

    auto result1_val = read_vector<float>(result1);
    EXPECT_TRUE(all_close_f(std::vector<float>{3, 5, 7}, result1_val));

    auto result2_val = read_vector<float>(result2);
    EXPECT_TRUE(all_close_f(std::vector<float>{1, 2, 3}, result2_val));
}

TEST(pre_post_process, mean_scale_vector_tensor_layout) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    auto name = f->get_parameters().front()->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    f = PrePostProcessor()
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_layout("NC??"))
                       .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
            .build(f);
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_TRUE(f->get_parameters().front()->has_layout());
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result}, {make_host_tensor<ngraph::element::f32>(Shape{1, 3, 2, 1}, {5., 1., 5., 11., 11., -1.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{2., 0., 1., 3., 2., -1.}, result_val));
}

TEST(pre_post_process, mean_scale_dynamic_layout) {
    auto f = create_simple_function(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3});
    auto name = f->get_parameters().front()->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    f = PrePostProcessor()
            .input(InputInfo()
                       .tensor(InputTensorInfo().set_layout("N...C"))
                       .preprocess(PreProcessSteps().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f})))
            .build(f);

    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_TRUE(f->get_parameters().front()->has_layout());
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "N...C");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);

    auto result = std::make_shared<HostTensor>();
    f->evaluate({result}, {make_host_tensor<ngraph::element::f32>(Shape{1, 2, 1, 3}, {5., 2., 7., 7., 8., -1.})});
    auto result_val = read_vector<float>(result);
    EXPECT_TRUE(all_close_f(std::vector<float>{2., 0., 1., 3., 2., -1.}, result_val));
}

TEST(pre_post_process, scale_vector_no_channels_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_THROW(f = PrePostProcessor()
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("N?HW"))
                                    .preprocess(PreProcessSteps().scale({0.1f, 0.2f, 0.3f})))
                         .build(f),
                 ov::AssertFailure);
}

TEST(pre_post_process, scale_vector_channels_out_of_range) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_THROW(f = PrePostProcessor()
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("0123C"))
                                    .preprocess(PreProcessSteps().scale({0.1f, 0.2f, 0.3f})))
                         .build(f),
                 ov::AssertFailure);
}

TEST(pre_post_process, mean_vector_no_layout) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_THROW(
        f = PrePostProcessor().input(InputInfo().preprocess(PreProcessSteps().mean({0.1f, 0.2f, 0.3f}))).build(f),
        ov::AssertFailure);
}

TEST(pre_post_process, mean_vector_dynamic_channels_shape) {
    auto f = create_simple_function(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    ASSERT_THROW(f = PrePostProcessor()
                         .input(InputInfo()
                                    .tensor(InputTensorInfo().set_layout("NCHW"))
                                    .preprocess(PreProcessSteps().mean({0.1f, 0.2f, 0.3f})))
                         .build(f),
                 ov::AssertFailure);
}
