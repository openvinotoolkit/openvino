// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/core/except.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/util/common_util.hpp"
#include "preprocess/color_utils.hpp"

using namespace ov;
using namespace ov::preprocess;

static std::shared_ptr<Model> create_simple_function(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto op = std::make_shared<op::v0::Relu>(data1);
    op->set_friendly_name("Relu");
    op->get_output_tensor(0).set_names({"tensor_Relu"});
    auto res = std::make_shared<op::v0::Result>(op);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1"});
    return std::make_shared<Model>(ResultVector{res}, ParameterVector{data1});
}

static std::shared_ptr<Model> create_trivial(element::Type type, const PartialShape& shape) {
    auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->get_output_tensor(0).set_names({"tensor_input1"});
    auto res = std::make_shared<op::v0::Result>(data1);
    res->set_friendly_name("Result1");
    res->get_output_tensor(0).set_names({"tensor_output1"});
    return std::make_shared<Model>(ResultVector{res}, ParameterVector{data1});
}

template <int N>
static std::shared_ptr<Model> create_n_inputs(element::Type type, const PartialShape& shape) {
    ResultVector res;
    ParameterVector params;
    for (size_t i = 0; i < N; i++) {
        auto index_str = std::to_string(i);
        auto data1 = std::make_shared<op::v0::Parameter>(type, shape);
        data1->set_friendly_name("input" + index_str);
        data1->get_output_tensor(0).set_names({"tensor_input" + index_str});
        auto op1 = std::make_shared<op::v0::Relu>(data1);
        op1->set_friendly_name("Relu" + index_str);
        auto res1 = std::make_shared<op::v0::Result>(op1);
        res1->set_friendly_name("Result" + index_str);
        res1->get_output_tensor(0).set_names({"tensor_output" + index_str});
        params.push_back(data1);
        res.push_back(res1);
    }
    return std::make_shared<Model>(res, params);
}

TEST(pre_post_process, simple_mean_scale) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().preprocess().mean(1.f).scale(2.f);
    f = p.build();
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, simple_mean_scale_getters_f16) {
    auto f = create_simple_function(element::f16, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input("tensor_input1").preprocess().mean(1).scale(2);
    f = p.build();
    EXPECT_EQ(f->get_output_element_type(0), element::f16);
}

TEST(pre_post_process, simple_mean_scale_getters_f64) {
    auto f = create_simple_function(element::f64, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input("tensor_input1").preprocess().mean(1).scale(2);
    f = p.build();
    EXPECT_EQ(f->get_output_element_type(0), element::f64);
}

TEST(pre_post_process, convert_element_type_and_scale) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::i16);
    p.input().preprocess().convert_element_type(element::f32).scale(2.f).convert_element_type(element::i8);
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i16);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, convert_element_type_implicit) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::i32);
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::f32);
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::i32);
}

TEST(pre_post_process, convert_element_type_implicit_several_time) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::i32);
    PrePostProcessor preprocessor(f);
    preprocessor.input().tensor().set_layout(ov::Layout("NHWC"));
    preprocessor.input().model().set_layout(ov::Layout("NCHW"));
    preprocessor.input().tensor().set_element_type(element::f16);
    preprocessor.input().tensor().set_element_type(element::i32);
    preprocessor.input().tensor().set_element_type(element::u32);
    preprocessor.input().tensor().set_element_type(element::f32);
    preprocessor.output().tensor().set_element_type(element::f16);
    preprocessor.output().tensor().set_element_type(element::i32);
    preprocessor.output().tensor().set_element_type(element::u32);
    preprocessor.output().tensor().set_element_type(element::f32);
    preprocessor.output().tensor().set_element_type(element::u64);
    f = preprocessor.build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_layout().to_string(), "[N,H,W,C]");
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::u64);
}

TEST(pre_post_process, convert_element_type_same) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    auto old_size = f->get_ops().size();
    auto p = PrePostProcessor(f);
    p.input("tensor_input1").tensor().set_element_type(element::i32);
    p.input("tensor_input1").preprocess().convert_element_type(element::i32);
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(old_size, f->get_ops().size());
}

TEST(pre_post_process, convert_element_type_default) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto type_custom1 = element::Type();
    auto type_custom2 = element::Type();
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::i32);
    p.input()
        .preprocess()
        .custom([&type_custom1](const Output<Node>& node) {
            type_custom1 = node.get_element_type();
            return node;
        })
        .convert_element_type()
        .custom([&type_custom2](const Output<Node>& node) {
            type_custom2 = node.get_element_type();
            return node;
        });
    f = p.build();
    EXPECT_EQ(type_custom1, element::i32);
    EXPECT_EQ(type_custom2, element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::f32);
}

TEST(pre_post_process, empty_preprocess) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::i8);
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i8);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
}

TEST(pre_post_process, preprocess_assert_input_without_index) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_ANY_THROW(p.input().preprocess().mean(0.f); f = p.build());
    EXPECT_ANY_THROW(p.input("some_non_existing_name").preprocess().mean(0.f); f = p.build());
}

TEST(pre_post_process, convert_element_type_from_unknown) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    ASSERT_THROW(p.input().preprocess().convert_element_type(element::dynamic).convert_element_type(element::i32);
                 f = p.build();
                 , ov::AssertFailure);
}

TEST(pre_post_process, scale_not_float) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    ASSERT_THROW(p.input().preprocess().convert_element_type(element::i32).scale(2.0f);
                 f = p.build(), ov::AssertFailure);
}

TEST(pre_post_process, mean_not_float) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    ASSERT_THROW(p.input().preprocess().convert_element_type(element::i32).mean(2.0f);
                 f = p.build(), ov::AssertFailure);
}

TEST(pre_post_process, tensor_element_type_and_scale) {
    auto f = create_simple_function(element::i8, Shape{1, 3, 1, 1});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::f32);
    p.input().preprocess().scale(2.0f).convert_element_type(element::i8);
    f = p.build();

    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_output_element_type(0), element::i8);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), Layout());
    EXPECT_EQ(ov::layout::get_layout(f->input(0)), Layout());
}

class convert_color_to_gray : public ::testing::TestWithParam<ColorFormat> {};

TEST_P(convert_color_to_gray, nhwc_to_nhwc) {
    const auto& COLOR_FORMAT = GetParam();

    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 1});
    auto p = PrePostProcessor(f);
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    p.input().tensor().set_color_format(COLOR_FORMAT);
    p.input().preprocess().convert_color(ColorFormat::GRAY);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(ov::layout::get_layout(f->input(0)), "NHWC");
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 2, 2, 3}));
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2, 2, 1}));
}

TEST_P(convert_color_to_gray, nchw_to_nchw) {
    const auto& COLOR_FORMAT = GetParam();

    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 1, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NCHW").set_color_format(COLOR_FORMAT);
    p.input().model().set_layout("NCHW");
    p.input().preprocess().convert_color(ColorFormat::GRAY);
    f = p.build();

    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NCHW");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 2}));
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1, 2, 2}));
}

TEST_P(convert_color_to_gray, nhwc_to_nchw) {
    const auto& COLOR_FORMAT = GetParam();

    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 1, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NHWC").set_color_format(COLOR_FORMAT);
    p.input().model().set_layout("NCHW");
    p.input().preprocess().convert_color(ColorFormat::GRAY);
    f = p.build();

    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 2, 2, 3}));
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 1, 2, 2}));
}

TEST_P(convert_color_to_gray, nchw_to_nhwc) {
    const auto& COLOR_FORMAT = GetParam();

    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 1});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NCHW").set_color_format(COLOR_FORMAT);
    p.input().model().set_layout("NHWC");
    p.input().preprocess().convert_color(ColorFormat::GRAY);
    f = p.build();

    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NCHW");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 2}));
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{Dimension::dynamic(), 2, 2, 1}));
}

TEST_P(convert_color_to_gray, assert_no_N_dim) {
    const auto& COLOR_FORMAT = GetParam();

    OV_EXPECT_THROW(auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 1});
                    auto p = PrePostProcessor(f);
                    p.input().tensor().set_layout("DHWC").set_color_format(COLOR_FORMAT);
                    p.input().preprocess().convert_color(ColorFormat::GRAY);
                    p.build();
                    , ov::AssertFailure, ::testing::HasSubstr("Dimension name 'N' is not found in layout"));
}

TEST_P(convert_color_to_gray, assert_no_C_dim) {
    const auto& COLOR_FORMAT = GetParam();

    OV_EXPECT_THROW(auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 1});
                    auto p = PrePostProcessor(f);
                    p.input().tensor().set_layout("NHWD").set_color_format(COLOR_FORMAT);
                    p.input().preprocess().convert_color(ColorFormat::GRAY);
                    p.build();
                    , ov::AssertFailure, ::testing::HasSubstr("C dimension index is not defined"));
}

TEST_P(convert_color_to_gray, assert_rank_less_4) {
    const auto& COLOR_FORMAT = GetParam();

    OV_EXPECT_THROW(auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 3});
                    auto p = PrePostProcessor(f);
                    p.input().tensor().set_layout("NHC").set_color_format(COLOR_FORMAT);
                    p.input().preprocess().convert_color(ColorFormat::GRAY);
                    p.build();
                    , ov::AssertFailure, ::testing::HasSubstr("Input shape size should be equal to 4, actual size: 3"));
}

TEST_P(convert_color_to_gray, assert_rank_more_4) {
    const auto& COLOR_FORMAT = GetParam();

    OV_EXPECT_THROW(auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 2, 3});
                    auto p = PrePostProcessor(f);
                    p.input().tensor().set_layout("NDHWC").set_color_format(COLOR_FORMAT);
                    p.input().preprocess().convert_color(ColorFormat::GRAY);
                    p.build();
                    , ov::AssertFailure, ::testing::HasSubstr("Input shape size should be equal to 4, actual size: 5"));
}

TEST_P(convert_color_to_gray, assert_C_not_equal_1) {
    const auto& COLOR_FORMAT = GetParam();

    OV_EXPECT_THROW(auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 5});
                    auto p = PrePostProcessor(f);
                    p.input().tensor().set_layout("NHWC").set_color_format(COLOR_FORMAT);
                    p.input().preprocess().convert_color(ColorFormat::GRAY);
                    p.build();
                    ,
                    ov::AssertFailure,
                    ::testing::HasSubstr("Resulting shape '[?,2,2,1]' after preprocessing is not aligned with original "
                                         "parameter's shape: [?,2,2,5]"));
}

INSTANTIATE_TEST_SUITE_P(pre_post_process,
                         convert_color_to_gray,
                         ::testing::Values(ColorFormat::RGB, ColorFormat::BGR),
                         [](const ::testing::TestParamInfo<convert_color_to_gray::ParamType>& info) {
                             std::string name =
                                 color_format_name(info.param) + "_to_" + color_format_name(ColorFormat::GRAY);
                             return name;
                         });

TEST(pre_post_process, convert_color_nv12_rgb_single) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    auto p = PrePostProcessor(f);
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    p.input().tensor().set_element_type(element::u8).set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB).convert_element_type(element::f32);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::u8);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(ov::layout::get_layout(f->input(0)), "NHWC");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
}

TEST(pre_post_process, convert_color_nv12_bgr_single) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters().front()->get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
}

TEST(pre_post_process, convert_color_nv12_bgr_2_planes) {
    auto f = create_simple_function(element::f32, Shape{5, 2, 2, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES, {"TestY", "TestUV"});
    p.input().preprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_friendly_name(), "input1/TestY");
    EXPECT_EQ(*f->get_parameters()[0]->output(0).get_tensor().get_names().begin(), "tensor_input1/TestY");
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{5, 2, 2, 1}));

    EXPECT_EQ(f->get_parameters()[1]->get_friendly_name(), "input1/TestUV");
    EXPECT_EQ(*f->get_parameters()[1]->output(0).get_tensor().get_names().begin(), "tensor_input1/TestUV");
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (PartialShape{5, 1, 1, 2}));
}

TEST(pre_post_process, convert_color_nv12_rgb_2_planes) {
    auto f = create_simple_function(element::f32, Shape{5, 2, 2, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{5, 2, 2, 1}));
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (PartialShape{5, 1, 1, 2}));
}

TEST(pre_post_process, convert_color_nv12_bgr_2_planes_u8) {
    auto f = create_simple_function(element::u8, Shape{1, 2, 2, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::u8);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{1, 2, 2, 1}));
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::u8);
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (PartialShape{1, 1, 1, 2}));
}

TEST(pre_post_process, convert_color_nv12_bgr_2_planes_el_type) {
    auto f = create_simple_function(element::u8, Shape{1, 2, 2, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::f32).set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_element_type(element::u8).convert_color(ColorFormat::BGR);
    f = p.build();

    ASSERT_EQ(f->get_parameters().size(), 2);
    EXPECT_EQ(f->get_parameters()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters()[1]->get_element_type(), element::f32);
}

TEST(pre_post_process, convert_color_i420_bgr_single) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->input().get_tensor().get_names();
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->inputs().size(), 1);
    EXPECT_EQ(f->input().get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->input().get_tensor().get_names(), tensor_names);
}

TEST(pre_post_process, convert_color_i420_rgb_single) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 4, 4, 3});
    auto name = f->get_parameters()[0]->get_friendly_name();
    auto tensor_names = f->input().get_tensor().get_names();
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    f = p.build();

    EXPECT_EQ(f->inputs().size(), 1);
    EXPECT_EQ(f->input().get_element_type(), element::f32);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NHWC");
    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{Dimension::dynamic(), 6, 4, 1}));
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->input().get_tensor().get_names(), tensor_names);
}

TEST(pre_post_process, convert_color_i420_bgr_3_planes) {
    auto f = create_simple_function(element::f32, Shape{5, 30, 20, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES, {"TestY", "TestU", "TestV"});
    p.input().preprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 3);

    EXPECT_EQ(f->get_parameters()[0]->get_friendly_name(), "input1/TestY");
    EXPECT_EQ(*f->input(0).get_tensor().get_names().begin(), "tensor_input1/TestY");
    EXPECT_EQ(f->input(0).get_element_type(), element::f32);
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{5, 30, 20, 1}));

    EXPECT_EQ(f->get_parameters()[1]->get_friendly_name(), "input1/TestU");
    EXPECT_EQ(*f->input(1).get_tensor().get_names().begin(), "tensor_input1/TestU");
    EXPECT_EQ(f->input(1).get_element_type(), element::f32);
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{5, 15, 10, 1}));

    EXPECT_EQ(f->get_parameters()[2]->get_friendly_name(), "input1/TestV");
    EXPECT_EQ(*f->input(2).get_tensor().get_names().begin(), "tensor_input1/TestV");
    EXPECT_EQ(f->input(2).get_element_type(), element::f32);
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{5, 15, 10, 1}));
}

TEST(pre_post_process, convert_color_i420_rgb_3_planes) {
    auto f = create_simple_function(element::u8, Shape{5, 20, 20, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    f = p.build();

    EXPECT_EQ(f->inputs().size(), 3);
    EXPECT_EQ(f->input(0).get_element_type(), element::u8);
    EXPECT_EQ(f->input(1).get_element_type(), element::u8);
    EXPECT_EQ(f->input(2).get_element_type(), element::u8);
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{5, 20, 20, 1}));
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{5, 10, 10, 1}));
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{5, 10, 10, 1}));
}

TEST(pre_post_process, convert_color_same_type) {
    auto f = create_simple_function(element::u8, Shape{1, 2, 2, 3});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_color_format(ColorFormat::RGB);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    f = p.build();

    EXPECT_EQ(f->get_parameters().size(), 1);
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, convert_color_unsupported) {
    // Feel free to update this test when more color conversions are supported in future
    auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});

    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
                 p.input().preprocess().convert_color(ColorFormat::UNDEFINED);
                 f = p.build(), ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
                 p.input().preprocess().convert_color(ColorFormat::UNDEFINED);
                 f = p.build(), ov::AssertFailure);

    auto colors = {ColorFormat::NV12_TWO_PLANES, ColorFormat::NV12_SINGLE_PLANE, ColorFormat::RGB, ColorFormat::BGR};
    for (const auto& color : colors) {
        EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::UNDEFINED);
                     p.input().preprocess().convert_color(color);
                     f = p.build(), ov::AssertFailure);

        EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(color);
                     p.input().preprocess().convert_color(ColorFormat::UNDEFINED);
                     f = p.build(), ov::AssertFailure);
    }
}

TEST(pre_post_process, convert_color_incorrect_subnames) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 2, 2, 3});
    EXPECT_THROW(auto p = PrePostProcessor(f);
                 p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE, {"Test"});
                 p.input().preprocess().convert_color(ColorFormat::RGB);
                 p.build(), ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE, {"Test"});
            p.input().preprocess().convert_color(ColorFormat::RGB);
            p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f);
                 p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Test"});
                 p.build(), ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES, {"Test"});
            p.input().preprocess().convert_color(ColorFormat::BGR);
            p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f);
                 p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES, {"1", "2", "3"});
                 f = p.build(), ov::AssertFailure);
}

TEST(pre_post_process, convert_color_duplicate_subnames) {
    auto f = create_n_inputs<2>(element::f32, PartialShape{1, 2, 2, 3});
    f->get_parameters()[0]->get_output_tensor(0).set_names({"tensor_input1"});
    f->get_parameters()[1]->get_output_tensor(0).set_names({"tensor_input1/CustomUV"});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE, {"CustomY", "CustomUV"});
                 p.input().preprocess().convert_color(ColorFormat::RGB);
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, convert_color_duplicate_internal_subnames_mean) {
    auto f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
    for (int i = 0; i < 10; i++) {
        // Create preprocessing step several times (try to duplicate internal node names this way)
        EXPECT_NO_THROW(auto p = PrePostProcessor(f); p.input().preprocess().mean(0.1f); f = p.build());
        EXPECT_NO_THROW(auto p = PrePostProcessor(f); p.input().preprocess().scale(1.1f); f = p.build());
        EXPECT_NO_THROW(auto p = PrePostProcessor(f);
                        p.input().preprocess().convert_element_type(element::u8).convert_element_type(element::f32);
                        f = p.build());
    }
    f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
    for (int i = 0; i < 10; i++) {
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NHWC");
        p.input().preprocess().convert_layout("NCHW");
        p.input().model().set_layout("NHWC");
        f = p.build();
    }
    f = create_simple_function(element::f32, PartialShape{1, 2, 2, 3});
    auto p = PrePostProcessor(f);
    for (int i = 10; i < 20; i++) {
        p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR, i, i);
    }
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().tensor().set_spatial_static_shape(480, 640);
    p.input().model().set_layout("NHWC");
    EXPECT_NO_THROW(f = p.build());
}

TEST(pre_post_process, convert_layout_implicit_several_time) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_parameters().front()->get_element_type(), element::i32);
    EXPECT_EQ(f->get_results().front()->get_element_type(), element::i32);
    PrePostProcessor preprocessor(f);
    preprocessor.input().tensor().set_layout("NWHC");
    preprocessor.input().tensor().set_layout("CHWN");
    preprocessor.input().tensor().set_layout("NHCW");
    preprocessor.input().tensor().set_layout("NCHW");
    preprocessor.input().tensor().set_layout("NHWC");
    preprocessor.output().tensor().set_layout("NWHC");
    preprocessor.output().tensor().set_layout("CHWN");
    preprocessor.output().tensor().set_layout("NHCW");
    preprocessor.output().tensor().set_layout("NCHW");
    f = preprocessor.build();
    EXPECT_EQ(f->get_parameters().front()->get_layout(), Layout("[N,H,W,C]"));
    EXPECT_EQ(f->get_results().front()->get_layout(), Layout("[N,C,H,W]"));
}

TEST(pre_post_process, tensor_set_layout) {
    auto f = create_n_inputs<6>(element::f32, Shape{1, 3, 480, 640});
    PrePostProcessor preprocessor(f);
    preprocessor.input(0).tensor().set_layout("NCHW");
    preprocessor.input(0).preprocess().mean({1.0, 2.0, 3.0});

    preprocessor.input(1).tensor().set_layout("NHWC");
    preprocessor.input(1).preprocess().mean({1.0, 2.0, 3.0}).convert_layout("NCHW");

    preprocessor.input(2).tensor().set_layout("NHWC");
    preprocessor.input(2).model().set_layout("NCHW");

    preprocessor.input(3).model().set_layout("NCHW");

    preprocessor.input(4).tensor().set_layout("NHWC");
    // Model layout will be calculated as "HWCN" -> "3,2,0,1" = NCHW
    preprocessor.input(4)
        .preprocess()
        .mean({1.0, 2.0, 3.0})
        .convert_layout({3, 2, 1, 0})
        .convert_layout("HWCN")
        .convert_layout({3, 2, 0, 1});

    preprocessor.input(5).tensor().set_layout("NHWC");
    preprocessor.input(5)
        .preprocess()
        .mean({1.0, 2.0, 3.0})
        .convert_layout({3, 2, 1, 0})   // NHWC -> CWHN
        .convert_layout({3, 0, 2, 1});  // CWHN -> NCHW

    f = preprocessor.build();
    EXPECT_EQ(f->get_parameters()[0]->get_partial_shape(), (Shape{1, 3, 480, 640}));
    EXPECT_EQ(f->get_parameters()[1]->get_partial_shape(), (Shape{1, 480, 640, 3}));
    EXPECT_EQ(f->get_parameters()[2]->get_partial_shape(), (Shape{1, 480, 640, 3}));
    EXPECT_EQ(f->get_parameters()[3]->get_partial_shape(), (Shape{1, 3, 480, 640}));
    EXPECT_EQ(f->get_parameters()[4]->get_partial_shape(), (Shape{1, 480, 640, 3}));
    EXPECT_EQ(f->get_parameters()[5]->get_partial_shape(), (Shape{1, 480, 640, 3}));
}

TEST(pre_post_process, postprocess_set_model_layout) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 224, 224});
    PrePostProcessor p(f);
    p.output(0).model().set_layout("NCHW");
    p.output(0).postprocess().convert_layout("NHWC");

    p.output(1).model().set_layout("NCHW");

    f = p.build();
    EXPECT_EQ(f->get_results()[0]->get_shape(), (Shape{1, 224, 224, 3}));
    EXPECT_EQ(f->get_results()[1]->get_shape(), (Shape{1, 3, 224, 224}));
}

TEST(pre_post_process, unsupported_model_color_format) {
    auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
                 f = p.build(), ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
                 f = p.build(), ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
                 p.input().preprocess().convert_layout("NCHW").convert_color(ColorFormat::RGB);
                 f = p.build(), ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
                 p.input().preprocess().mean(0.1f).convert_color(ColorFormat::RGB);
                 f = p.build(), ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f); p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
                 p.input().preprocess().scale(2.1f).convert_color(ColorFormat::RGB);
                 f = p.build(), ov::AssertFailure);
}

TEST(pre_post_process, unsupported_model_color_format_i420) {
    auto f = create_simple_function(element::f32, PartialShape{1, 4, 4, 3});
    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
            f = p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
            f = p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
            p.input().preprocess().convert_layout("NCHW").convert_color(ColorFormat::RGB);
            f = p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
            p.input().preprocess().scale(2.1f).convert_color(ColorFormat::BGR);
            f = p.build();
        },
        ov::AssertFailure);
}

TEST(pre_post_process, custom_preprocessing) {
    auto f = create_simple_function(element::i32, Shape{1, 3, 1, 1});
    auto p = PrePostProcessor(f);
    p.input().preprocess().custom([](const Output<Node>& node) {
        return std::make_shared<op::v0::Abs>(node);
    });
    f = p.build();
    EXPECT_EQ(f->get_output_element_type(0), element::i32);
}

TEST(pre_post_process, test_2_inputs_basic) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 1, 1});
    auto p = PrePostProcessor(f);
    p.input(1).preprocess().mean(1.f).scale(2.0f);
    f = p.build();
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    EXPECT_EQ(f->get_output_element_type(1), element::f32);
}

TEST(pre_post_process, set_model_input_layout_helper) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    ov::layout::set_layout(f->input(0), "NCHW");
    EXPECT_EQ(ov::layout::get_layout(f->input(0)), "NCHW");
}

TEST(pre_post_process, set_model_output_layout_helper) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    ov::layout::set_layout(f->output(0), "NCHW");
    EXPECT_EQ(ov::layout::get_layout(f->output(0)), "NCHW");
}

TEST(pre_post_process, reuse_model_layout_no_tensor_info) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    f->get_parameters().front()->set_layout("NC??");
    auto p = PrePostProcessor(f);
    p.input().preprocess().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f});
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
}

TEST(pre_post_process, set_model_layout_when_already_exists) {
    auto m = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    {
        auto p = PrePostProcessor(m);
        p.input().model().set_layout("N???");
        m = p.build();
    }
    EXPECT_EQ(m->input().get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    {
        auto p = PrePostProcessor(m);
        p.input().tensor().set_layout("NHWC");
        p.input().model().set_layout("NCHW");  // Expect "N???" will be overwritten by "NCHW"
        m = p.build();
    }
    EXPECT_EQ(m->input().get_partial_shape(), (PartialShape{Dimension::dynamic(), 2, 1, 3}));
}

TEST(pre_post_process, set_layout_out_of_bounds) {
    auto shape = PartialShape{1, 2};
    std::stringstream shape_str;
    shape_str << shape;
    auto f = create_simple_function(element::f32, shape);
    Layout from{"NHWC"};
    Layout to{"NCHW"};
    // TODO: replace with EXPECT_THAT after upgrade gtest to v1.11
    try {
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout(from);
        p.input().model().set_layout(to);
        f = p.build();
        FAIL() << "Layout conversion shall throw";
    } catch (const ov::Exception& err) {
        // Verify that error message contains tensor and network  layout
        EXPECT_TRUE(std::string(err.what()).find(from.to_string()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(to.to_string()) != std::string::npos) << err.what();
        // Verify that error message contains 'shape' word
        EXPECT_TRUE(std::string(err.what()).find(shape_str.str()) != std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(pre_post_process, reuse_model_layout_tensor_info) {
    auto f = create_simple_function(element::u8, PartialShape{Dimension::dynamic(), 3, 2, 1});
    f->get_parameters().front()->set_layout("NC??");
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::f32);
    p.input().preprocess().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f}).convert_element_type(element::u8);
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
}

TEST(pre_post_process, mean_scale_vector_tensor_layout) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    auto name = f->get_parameters().front()->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NC??");
    p.input().preprocess().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f});
    f = p.build();
    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "NC??");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, mean_scale_dynamic_layout) {
    auto f = create_simple_function(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3});
    auto name = f->get_parameters().front()->get_friendly_name();
    auto tensor_names = f->get_parameters().front()->get_output_tensor(0).get_names();
    auto p = PrePostProcessor(f);

    p.input().tensor().set_layout("N...C");
    p.input().preprocess().mean({1.f, 2.f, 3.f}).scale({2.f, 3.f, 4.f});
    f = p.build();

    EXPECT_EQ(f->get_parameters().front()->get_friendly_name(), name);
    EXPECT_EQ(f->get_parameters().front()->get_layout(), "N...C");
    EXPECT_EQ(f->get_parameters().front()->get_output_tensor(0).get_names(), tensor_names);
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, scale_vector_no_channels_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_layout("N?HW"); p.input().preprocess().scale({0.1f, 0.2f, 0.3f});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, scale_vector_dim_mismatch) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_layout("NCHW"); p.input().preprocess().scale({0.1f, 0.2f, 0.3f, 0.4f});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, scale_vector_channels_out_of_range) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    auto p = PrePostProcessor(f);
    ASSERT_THROW(p.input().tensor().set_layout("0123C"); p.input().preprocess().scale({0.1f, 0.2f, 0.3f});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, mean_vector_no_layout) {
    auto f = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 224, 224});
    ASSERT_EQ(f->get_output_element_type(0), element::f32);
    auto p = PrePostProcessor(f);
    ASSERT_THROW(p.input().preprocess().mean({0.1f, 0.2f, 0.3f}); p.build(), ov::AssertFailure);
}

TEST(pre_post_process, mean_vector_dynamic_channels_shape) {
    auto f = create_simple_function(
        element::f32,
        PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
    auto p = PrePostProcessor(f);
    EXPECT_NO_THROW(p.input().tensor().set_layout("NCHW"); p.input().preprocess().mean({0.1f, 0.2f, 0.3f}); p.build());
    EXPECT_EQ(f->get_output_element_type(0), element::f32);
}

TEST(pre_post_process, pad_vector_constant_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 200, 200});
    auto p = PrePostProcessor(f);

    p.input().tensor().set_shape({1, 3, 199, 199});
    p.input().preprocess().pad({0, 0, 0, 0}, {0, 0, 1, 1}, 0, PaddingMode::CONSTANT);
    EXPECT_NO_THROW(p.build());
}

TEST(pre_post_process, pad_vector_out_of_range) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 5, 5});
    auto p = PrePostProcessor(f);

    ASSERT_THROW(p.input().preprocess().pad({0, 0, -2, 0}, {0, 0, -4, 1}, 0, PaddingMode::CONSTANT);
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, pad_vector_dim_mismatch) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 5, 5});
    auto p = PrePostProcessor(f);

    ASSERT_THROW(p.input().preprocess().pad({0, 0, 2, 0, 1}, {0, 0, 4, 1, 1}, 0, PaddingMode::CONSTANT);
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, resize_no_model_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NHWC");
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
    EXPECT_NO_THROW(p.build());
}

TEST(pre_post_process, resize_with_bilinear_pillow) {
    const auto f = create_simple_function(element::f32, PartialShape{1, 3, 224, 224});

    auto p = PrePostProcessor(f);
    // make the model accept images with spatial dimensions different than the original model's dims
    p.input().tensor().set_shape({1, 3, 100, 150}).set_layout("NCHW");
    // resize the incoming images to the original model's dims (deduced by PPP)
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_BILINEAR_PILLOW);
    p.build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 3, 100, 150}));
    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 3, 224, 224}));
}

TEST(pre_post_process, resize_with_bicubic_pillow) {
    const auto f = create_simple_function(element::f32, PartialShape{1, 3, 100, 100});

    auto p = PrePostProcessor(f);
    // make the model accept images with any spatial dimensions
    p.input().tensor().set_shape({1, 3, -1, -1}).set_layout("NCHW");
    // resize the incoming images to the original model's dims (specified manually)
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_BICUBIC_PILLOW, 100, 100);
    p.build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 3, -1, -1}));
    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 3, 100, 100}));
}

// Error cases for 'resize'
TEST(pre_post_process, tensor_spatial_shape_no_layout_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_layout("NC?W").set_spatial_static_shape(480, 640);
                 p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
                 p.build(), ov::AssertFailure);

    EXPECT_THROW(p.input().tensor().set_layout("NCH?").set_spatial_static_shape(480, 640);
                 p.input().preprocess().resize(ResizeAlgorithm::RESIZE_CUBIC);
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, tensor_set_shape_for_resize) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_shape({1, 720, 1280, 3}).set_layout("NHWC");
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    p.build();
    EXPECT_EQ(f->input().get_partial_shape(), (Shape{1, 720, 1280, 3}));
    EXPECT_EQ(f->output().get_partial_shape(), (Shape{1, 3, 224, 224}));
}

TEST(pre_post_process, tensor_set_shape_incompatible) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_shape({1, 4, 224, 224}); p.build(), ov::AssertFailure);
}

// Check that 'set_shape' shall not be used together with set_spatial_..._shape
// This test can be removed if this requirement is relaxed in future releases
TEST(pre_post_process, tensor_set_shape_with_spatial) {
    auto f = create_simple_function(element::f32, PartialShape{-1, -1, -1, -1});
    {
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        EXPECT_THROW(p.input().tensor().set_shape({1, 3, 224, 224}).set_spatial_static_shape(448, 448);
                     p.build(), ov::AssertFailure);
    }
    {
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        EXPECT_THROW(p.input().tensor().set_spatial_static_shape(448, 448).set_shape({1, 3, 224, 224});
                     p.build(), ov::AssertFailure);
    }
    {
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        EXPECT_THROW(p.input().tensor().set_shape({1, 3, 224, 224}).set_spatial_dynamic_shape();
                     p.build(), ov::AssertFailure);
    }
    {
        auto p = PrePostProcessor(f);
        p.input().tensor().set_layout("NCHW");
        EXPECT_THROW(p.input().tensor().set_spatial_dynamic_shape().set_shape({1, 3, 224, 224});
                     p.build(), ov::AssertFailure);
    }
}

TEST(pre_post_process, resize_no_tensor_height) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_layout("N?WC"); p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
                 p.input().model().set_layout("NHWC");
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, resize_no_tensor_width) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 224, 224});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_layout("NH?C"); p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
                 p.input().model().set_layout("NHWC");
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto name = f->get_results().front()->get_friendly_name();
    auto name_last_op = f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names = f->output().get_tensor().get_names();

    auto p = PrePostProcessor(f);

    p.input().tensor().set_layout("NHWC");
    p.input().model().set_layout("NCHW");
    p.build();
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_EQ(name, f->get_results().front()->get_friendly_name());
    EXPECT_EQ(name_last_op,
              f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(tensor_names, f->output().get_tensor().get_names());
}

TEST(pre_post_process, preprocess_convert_layout_default) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);

    p.input().tensor().set_layout("NHWC");
    p.input().preprocess().convert_layout();
    p.input().model().set_layout("NCHW");
    p.build();
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, preprocess_convert_layout_same_various) {
    for (size_t i = 1; i < 100; i++) {
        auto f = create_simple_function(element::f32, PartialShape::dynamic(static_cast<int64_t>(i)));
        auto p = PrePostProcessor(f);
        std::stringstream stream;
        stream << "[0";
        for (size_t j = 1; j < i; j++) {
            stream << "," << std::to_string(j);
        }
        stream << "]";
        auto l = stream.str();
        p.input().tensor().set_layout(ov::Layout(l));
        p.input().model().set_layout(ov::Layout(std::string(i, '?')));
        EXPECT_NO_THROW(p.build());
    }
}

TEST(pre_post_process, preprocess_convert_layout_same) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto size_old = f->get_ordered_ops().size();

    auto p = PrePostProcessor(f);

    p.input().tensor().set_layout("NCHW");
    p.input().preprocess().convert_layout("NCHW");
    p.input().model().set_layout("NCHW");
    p.build();
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_parameters()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    // Verify that redundant ops were not added
    EXPECT_EQ(size_old, f->get_ordered_ops().size());
}

TEST(pre_post_process, preprocess_convert_layout_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    auto p = PrePostProcessor(f);
    p.input().preprocess().convert_layout({0, 3, 1, 2});
    p.input().model().set_layout("NCHW");
    p.build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 480, 640, 3}));
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), Layout("NHWC"));
}

TEST(pre_post_process, preprocess_convert_layout_dims_empty) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    auto p = PrePostProcessor(f);

    p.input().preprocess().convert_layout(std::vector<uint64_t>{});
    p.build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 3, 480, 640}));
}

TEST(pre_post_process, preprocess_convert_layout_dims_dyn_shape) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());

    auto p = PrePostProcessor(f);
    p.input().preprocess().convert_layout({0, 3, 1, 2});
    p.build();

    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape::dynamic()));
}

TEST(pre_post_process, preprocess_convert_layout_invalid_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().preprocess().convert_layout({0, 3, 2, 2}); p.build(), ov::AssertFailure);

    EXPECT_THROW(p.input().preprocess().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_invalid_dims_dyn_shape) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().preprocess().convert_layout({0, 3, 2, 2}); p.build(), ov::AssertFailure);

    EXPECT_THROW(p.input().preprocess().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_partially_defined) {
    auto f = create_n_inputs<8>(element::f32, Shape{1, 2, 3, 4, 5});

    auto p = PrePostProcessor(f);
    p.input(0).tensor().set_layout("nc???");
    p.input(0).model().set_layout("????c");

    p.input(1).tensor().set_layout("...c??");
    p.input(1).model().set_layout("ndhwc");

    p.input(2).tensor().set_layout("?cwh...");
    p.input(2).model().set_layout("...hwc");

    p.input(3).tensor().set_layout("...c");
    p.input(3).model().set_layout("c...");

    p.input(4).tensor().set_layout("...");
    p.input(4).model().set_layout("c...");

    p.input(5).tensor().set_layout("...c");
    p.input(5).model().set_layout("...");

    p.input(6).tensor().set_layout("ndhwc");
    p.input(6).model().set_layout("ndh?c");

    p.input(7).tensor().set_layout("ndh?c");
    p.input(7).model().set_layout("ndhwc");

    f = p.build();
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{1, 5, 2, 3, 4}));
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 2, 5, 3, 4}));
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{1, 5, 4, 3, 2}));
    EXPECT_EQ(f->input(3).get_partial_shape(), (PartialShape{2, 3, 4, 5, 1}));
    EXPECT_EQ(f->input(4).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    EXPECT_EQ(f->input(5).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    EXPECT_EQ(f->input(6).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    EXPECT_EQ(f->input(7).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
}

TEST(pre_post_process, preprocess_convert_layout_partially_defined_trivial) {
    auto f = create_n_inputs<4>(element::f32, Shape{1, 2, 3, 4, 5});
    auto ops_num = f->get_ordered_ops().size();

    auto p = PrePostProcessor(f);
    p.input(0).tensor().set_layout("...");
    p.input(0).model().set_layout("c...");

    p.input(1).tensor().set_layout("...c");
    p.input(1).model().set_layout("...");

    p.input(2).tensor().set_layout("ndhwc");
    p.input(2).model().set_layout("ndh?c");

    p.input(3).tensor().set_layout("ndh?c");
    p.input(3).model().set_layout("ndhwc");

    f = p.build();
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    EXPECT_EQ(f->input(3).get_partial_shape(), (PartialShape{1, 2, 3, 4, 5}));
    // Verify that no preprocessing Nodes are inserted
    EXPECT_EQ(ops_num, f->get_ordered_ops().size());
}

TEST(pre_post_process, preprocess_convert_layout_squeeze) {
    auto f = create_n_inputs<3>(element::f32, Shape{1, 3, 1, 480, 640});
    auto p = PrePostProcessor(f);

    p.input(0).tensor().set_layout("HWC");
    p.input(0).model().set_layout("NCDHW");

    p.input(1).tensor().set_layout("NHWC");
    p.input(1).model().set_layout("NCDHW");

    p.input(2).tensor().set_layout("WCHD");
    p.input(2).model().set_layout("NCDHW");

    p.build();
    EXPECT_EQ(ov::layout::get_layout(f->input(0)), "HWC");
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{480, 640, 3}));
    EXPECT_EQ(ov::layout::get_layout(f->input(1)), "NHWC");
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 480, 640, 3}));
    EXPECT_EQ(ov::layout::get_layout(f->input(2)), "WCHD");
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{640, 3, 480, 1}));
}

TEST(pre_post_process, preprocess_convert_layout_squeeze_dynamic) {
    auto f = create_n_inputs<2>(element::f32, PartialShape{Dimension::dynamic(), 3, 1, 480, 640});
    auto p = PrePostProcessor(f);

    p.input(0).tensor().set_layout("HWC");
    p.input(0).model().set_layout("NCDHW");

    p.input(1).tensor().set_layout("NHWC");
    p.input(1).model().set_layout("NCDHW");

    p.build();
    EXPECT_EQ(ov::layout::get_layout(f->input(0)), "HWC");
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{480, 640, 3}));
    EXPECT_EQ(ov::layout::get_layout(f->input(1)), "NHWC");
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{Dimension::dynamic(), 480, 640, 3}));
}

TEST(pre_post_process, preprocess_convert_layout_squeeze_unsupported) {
    auto f = create_n_inputs<1>(element::f32, PartialShape{Dimension::dynamic(), 3, 1, 480, 640});
    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input(0).tensor().set_layout("NCDHWS");
            p.input(0).model().set_layout("NCDHW");
            p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input(0).tensor().set_layout("HWC");
            p.input(0).model().set_layout("?????");
            p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input(0).tensor().set_layout("...S");
            p.input(0).model().set_layout("NCDHW");
            p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input(0).tensor().set_layout("HWC");
            p.input(0).model().set_layout("...NCDHW");
            p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input(0).tensor().set_layout("HW?");
            p.input(0).model().set_layout("NCDHW");
            p.build();
        },
        ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_partially_defined_error) {
    auto f = create_simple_function(element::f32, Shape{1, 2, 3, 4, 5});

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_layout("nch??");
            p.input().model().set_layout("???wc");
            f = p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_layout("nch??");
            p.input().model().set_layout("???wc?");
            f = p.build();
        },
        ov::AssertFailure);
}

TEST(pre_post_process, preprocess_convert_layout_partially_defined_error_diff_rank) {
    auto f = create_simple_function(element::f32, Shape{1, 2, 3, 4, 5});
}

TEST(pre_post_process, preprocess_convert_layout_partially_defined_error_dyn_rank) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_layout("nchw");
            p.input().model().set_layout("...wc");
            f = p.build();
        },
        ov::AssertFailure);

    EXPECT_THROW(
        {
            auto p = PrePostProcessor(f);
            p.input().tensor().set_layout("nchw");
            p.input().model().set_layout("??wc?");
            f = p.build();
        },
        ov::AssertFailure);
}

TEST(pre_post_process, preprocess_reverse_channels_multiple_planes) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(

        p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Y", "UV"});
        p.input().preprocess().reverse_channels();
        p.build(), ov::AssertFailure);
}

TEST(pre_post_process, preprocess_reverse_channels_no_c_dim) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.input().tensor().set_layout("N?HW"); p.input().preprocess().reverse_channels();
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, preprocess_reverse_channels_no_shape_inference) {
    auto f = create_simple_function(element::f32,
                                    PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()});
    auto out_shape = f->output(0).get_partial_shape();

    using namespace ov::preprocess;
    PrePostProcessor p(f);
    p.input(0).tensor().set_layout("NCHW");
    p.input(0).preprocess().reverse_channels();
    OV_ASSERT_NO_THROW(p.build());
    // Ensure that {?,3,?,?} is not transformed to {?,?,?,?}
    EXPECT_EQ(out_shape, f->output(0).get_partial_shape());
}

TEST(pre_post_process, preprocess_preserve_rt_info) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f->get_parameters()[0]->get_rt_info()["someKey"] = "someValue";
    f->input().get_rt_info()["someKey_in"] = "someValue_in";
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::u8);
    f = p.build();
    EXPECT_EQ(f->input().get_element_type(), element::u8);

    ASSERT_EQ(f->get_parameters()[0]->get_rt_info().count("someKey"), 1);
    auto var0 = f->get_parameters()[0]->get_rt_info()["someKey"].as<std::string>();
    EXPECT_EQ(var0, "someValue");

    ASSERT_EQ(f->input().get_rt_info().count("someKey_in"), 1);
    auto var0_in = f->input().get_rt_info()["someKey_in"].as<std::string>();
    EXPECT_EQ(var0_in, "someValue_in");
}

TEST(pre_post_process, preprocess_memory_type) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_memory_type("abc");
    f = p.build();
    ASSERT_EQ(f->input().get_rt_info().count(TensorInfoMemoryType::get_type_info_static()), 1);
    auto var0 = f->input().get_rt_info()[TensorInfoMemoryType::get_type_info_static()].as<TensorInfoMemoryType>().value;
    EXPECT_EQ(var0, "abc");
}

TEST(pre_post_process, preprocess_memory_type_clear) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f->input().get_rt_info()[TensorInfoMemoryType::get_type_info_static()] = TensorInfoMemoryType("abc");
    auto p = PrePostProcessor(f);
    p.input().tensor().set_memory_type("");
    f = p.build();
    EXPECT_EQ(f->input().get_rt_info().count(TensorInfoMemoryType::get_type_info_static()), 0);
}

TEST(pre_post_process, preprocess_memory_type_not_cleared) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_memory_type("abc").set_layout("NHWC");
    f = p.build();

    ASSERT_EQ(f->input().get_rt_info().count(TensorInfoMemoryType::get_type_info_static()), 1);
    auto var0 = f->input().get_rt_info()[TensorInfoMemoryType::get_type_info_static()].as<TensorInfoMemoryType>().value;
    EXPECT_EQ(var0, "abc");
}

TEST(pre_post_process, preprocess_from) {
    auto t = ov::Tensor(element::u8, {1, 480, 640, 3});
    auto f = create_simple_function(element::f32, Shape{1, 224, 224, 3});
    ov::layout::set_layout(f->input(), "NHWC");
    auto p = PrePostProcessor(f);
    p.input().tensor().set_from(t);
    p.input().preprocess().resize(ResizeAlgorithm::RESIZE_LINEAR);
    f = p.build();

    EXPECT_EQ(f->input().get_element_type(), element::u8);
    EXPECT_EQ(f->input().get_shape(), (Shape{1, 480, 640, 3}));
    EXPECT_EQ(f->output().get_element_type(), element::f32);
    EXPECT_EQ(f->output().get_shape(), (Shape{1, 224, 224, 3}));
}

TEST(pre_post_process, preprocess_crop) {
    auto model = create_n_inputs<1>(element::f32, PartialShape::dynamic());
    auto p = PrePostProcessor(model);

    p.input().tensor().set_shape(Shape{1, 3, 200, 400});
    auto begin = std::vector<int>{0, 0, 50, 100};
    auto end = std::vector<int>{1, 3, 150, 300};
    auto begin_dump = ov::util::vector_to_string(begin);
    auto end_dump = ov::util::vector_to_string(end);
    p.input().preprocess().crop(begin, end);

    std::stringstream dump;
    dump << p;
    EXPECT_TRUE(dump.str().find(begin_dump) != std::string::npos)
        << "Dump doesn't contain begin coordinate. " << dump.str() << begin_dump;
    EXPECT_TRUE(dump.str().find(end_dump) != std::string::npos)
        << "Dump doesn't contain end coordinate. " << dump.str() << end_dump;
    p.build();

    // Verify that output will be {1, 3, 100, 200}
    EXPECT_EQ(model->output().get_partial_shape(), (PartialShape{1, 3, 100, 200}));
}

TEST(pre_post_process, preprocess_crop_wrong_dims) {
    auto model = create_n_inputs<1>(element::f32, PartialShape::dynamic());
    auto p = PrePostProcessor(model);

    p.input().tensor().set_shape(Shape{1, 3, 200, 400});
    auto begin = std::vector<int>{0, 0, 50, 100};
    auto end = std::vector<int>{1, 3, 150};
    auto begin_dump = ov::util::vector_to_string(begin);
    auto end_dump = ov::util::vector_to_string(end);
    try {
        p.input().preprocess().crop(begin, end);
        FAIL() << "crop with wrong dims shall throw";
    } catch (const ov::Exception& err) {
        EXPECT_TRUE(std::string(err.what()).find(begin_dump) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(end_dump) != std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

TEST(pre_post_process, preprocess_crop_wrong_dims_not_aligned) {
    auto model = create_n_inputs<1>(element::f32, PartialShape{1, 3, 100, 200});
    auto p = PrePostProcessor(model);

    p.input().tensor().set_shape(Shape{1, 3, 200});
    auto begin = std::vector<int>{0, 0, 50};
    auto end = std::vector<int>{1, 3, 150};
    std::stringstream exp_dump, act_dump;
    exp_dump << model->input().get_partial_shape();
    act_dump << PartialShape{1, 3, 100};
    try {
        p.input().preprocess().crop(begin, end);
        p.build();
        FAIL() << "crop with wrong dims rank shall throw";
    } catch (const ov::Exception& err) {
        EXPECT_TRUE(std::string(err.what()).find(exp_dump.str()) != std::string::npos) << err.what();
        EXPECT_TRUE(std::string(err.what()).find(act_dump.str()) != std::string::npos) << err.what();
    } catch (...) {
        FAIL() << "Expected ov::Exception";
    }
}

// --- PostProcess - set/convert element type ---

TEST(pre_post_process, postprocess_convert_element_type_explicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto name = f->output().get_node_shared_ptr()->get_friendly_name();
    auto name_last_op = f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
    auto old_names = f->output().get_tensor().get_names();
    auto p = PrePostProcessor(f);

    p.output().postprocess().convert_element_type(element::u8);
    p.build();
    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::u8);
    EXPECT_EQ(f->output().get_tensor().get_names(), old_names);
    EXPECT_EQ(old_names.count("tensor_output1"), 1);
    auto ops = f->get_ordered_ops();
    auto res_count = std::count_if(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& n) {
        return std::dynamic_pointer_cast<ov::op::v0::Result>(n) != nullptr;
    });
    EXPECT_EQ(res_count, 1);
    auto names_count = std::count_if(ops.begin(), ops.end(), [](std::shared_ptr<ov::Node> n) {
        return n->output(0).get_tensor().get_names().count("tensor_output1") > 0;
    });
    EXPECT_EQ(names_count, 2);  // last node + result referencing to it
    EXPECT_EQ(name, f->output().get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(name_last_op,
              f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());
}

TEST(pre_post_process, postprocess_convert_element_type_default) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto name = f->output(1).get_node_shared_ptr()->get_friendly_name();
    auto name_last_op = f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names = f->output(1).get_tensor().get_names();
    auto p = PrePostProcessor(f);

    p.output(1).postprocess().convert_element_type();
    p.output(1).tensor().set_element_type(element::u8);
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::f32);
    EXPECT_EQ(f->get_results()[1]->get_element_type(), element::u8);
    EXPECT_EQ(name, f->output(1).get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(name_last_op,
              f->get_results().front()->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());
    EXPECT_EQ(tensor_names, f->output(1).get_tensor().get_names());
}

TEST(pre_post_process, postprocess_convert_element_type_same) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto size_old = f->get_ordered_ops().size();
    auto p = PrePostProcessor(f);

    p.output("tensor_output1").postprocess().convert_element_type(element::f32);
    p.output("tensor_output1").tensor().set_element_type(element::f32);
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::f32);

    // Verify that redundant ops were not added
    EXPECT_EQ(size_old, f->get_ordered_ops().size());
}

TEST(pre_post_process, postprocess_convert_element_type_default_error) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.output().postprocess().convert_element_type(); p.build(), ov::AssertFailure);
}

TEST(pre_post_process, postprocess_convert_element_type_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.output().tensor().set_element_type(element::u8);
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::u8);
}

TEST(pre_post_process, preprocess_keep_params_order) {
    auto f = create_n_inputs<3>(element::f32, Shape{1, 2, 2, 3});
    auto p = PrePostProcessor(f);

    p.input(1).tensor().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Y", "UV"});
    p.input(1).preprocess().convert_color(ColorFormat::RGB);
    p.input(0).tensor().set_layout("NCHW");
    p.input(2).tensor().set_color_format(ColorFormat::NV12_TWO_PLANES, {"Y", "UV"});
    p.input(2).preprocess().convert_color(ColorFormat::RGB);
    p.build();
    ASSERT_EQ(f->get_parameters().size(), 5);
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_parameters()[1]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[2]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[3]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_parameters()[4]->get_layout(), "NHWC");

    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 2, 2, 1}));
    EXPECT_EQ(f->input(2).get_partial_shape(), (PartialShape{1, 1, 1, 2}));
    EXPECT_EQ(f->input(3).get_partial_shape(), (PartialShape{1, 2, 2, 1}));
    EXPECT_EQ(f->input(4).get_partial_shape(), (PartialShape{1, 1, 1, 2}));

    EXPECT_EQ(f->input(0).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input0"});
    EXPECT_EQ(f->input(1).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input1/Y"});
    EXPECT_EQ(f->input(2).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input1/UV"});
    EXPECT_EQ(f->input(3).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input2/Y"});
    EXPECT_EQ(f->input(4).get_tensor().get_names(), std::unordered_set<std::string>{"tensor_input2/UV"});
}

// --- PostProcess - set/convert layout ---
TEST(pre_post_process, postprocess_set_layout_model) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NCHW");
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NCHW");
    EXPECT_EQ(ov::layout::get_layout(f->output(0)), "NCHW");
}

TEST(pre_post_process, postprocess_convert_layout_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    auto p = PrePostProcessor(f);

    p.output().model().set_layout("NCHW");
    p.output().tensor().set_layout("NHWC");
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_set_model_layout_when_already_exists) {
    auto m = create_simple_function(element::f32, PartialShape{Dimension::dynamic(), 3, 2, 1});
    {
        auto p = PrePostProcessor(m);
        p.output().model().set_layout("N???");
        m = p.build();
    }
    EXPECT_EQ(m->output().get_partial_shape(), (PartialShape{Dimension::dynamic(), 3, 2, 1}));
    {
        auto p = PrePostProcessor(m);
        p.output().model().set_layout("NCHW");  // Expect "N???" will be overwritten by "NCHW"
        p.output().tensor().set_layout("NHWC");
        m = p.build();
    }
    EXPECT_EQ(m->output().get_partial_shape(), (PartialShape{Dimension::dynamic(), 2, 1, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_explicit_no_target) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);

    p.output(1).model().set_layout("NCHW");
    p.output(1).postprocess().convert_layout("NHWC");
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    EXPECT_EQ(f->get_results()[1]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_default) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    auto p = PrePostProcessor(f);

    p.output().model().set_layout("NCHW");
    p.output().postprocess().convert_layout();
    p.output().tensor().set_layout("NHWC");
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_default_getters) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});

    auto p = PrePostProcessor(f);
    auto& out = p.output();
    out.model().set_layout("NCHW");
    out.postprocess().convert_layout();
    out.tensor().set_layout("NHWC");
    f = p.build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_same) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto size_old = f->get_ordered_ops().size();

    auto p = PrePostProcessor(f);

    p.output().model().set_layout("NCHW");
    p.output().postprocess().convert_layout("NCHW");
    p.output().tensor().set_layout("NCHW");
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    // Verify that redundant ops were not added
    EXPECT_EQ(size_old, f->get_ordered_ops().size());
}

TEST(pre_post_process, postprocess_convert_layout_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    auto p = PrePostProcessor(f);
    p.output().postprocess().convert_layout({0, 2, 3, 1});
    p.build();

    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 480, 640, 3}));
}

TEST(pre_post_process, postprocess_convert_layout_dims_empty) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    auto p = PrePostProcessor(f);

    p.output().postprocess().convert_layout(std::vector<uint64_t>{});
    p.build();

    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 3, 480, 640}));
}

TEST(pre_post_process, postprocess_convert_layout_has_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 480, 640});

    auto p = PrePostProcessor(f);

    p.output().model().set_layout("NC??");
    p.output().postprocess().convert_layout({0, 2, 3, 1});
    p.build();

    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 480, 640, 3}));
    EXPECT_EQ(f->get_results()[0]->get_layout(), "N??C");
}

TEST(pre_post_process, postprocess_convert_layout_invalid_dims) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.output().postprocess().convert_layout({0, 3, 2, 2}); p.build(), ov::AssertFailure);

    EXPECT_THROW(p.output().postprocess().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, postprocess_convert_layout_invalid_dims_dyn_shape) {
    auto f = create_simple_function(element::f32, PartialShape::dynamic());
    auto p = PrePostProcessor(f);
    EXPECT_THROW(p.output().postprocess().convert_layout({0, 3, 2, 2}); p.build(), ov::AssertFailure);

    EXPECT_THROW(p.output().postprocess().convert_layout({0, 3, 1, std::numeric_limits<uint64_t>::max()});
                 p.build(), ov::AssertFailure);
}

TEST(pre_post_process, postprocess_keep_friendly_names_compatibility) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 10, 10});
    auto result_fr_name = f->get_results()[0]->get_friendly_name();
    auto node_before_result_old = f->get_results()[0]->get_input_source_output(0).get_node_shared_ptr();
    auto node_name = node_before_result_old->get_friendly_name();
    auto p = PrePostProcessor(f);
    p.output().postprocess().convert_element_type(element::u8);
    f = p.build();
    EXPECT_EQ(f->get_results()[0]->get_friendly_name(), result_fr_name);
    auto node_before_result_new = f->get_results()[0]->get_input_source_output(0).get_node_shared_ptr();
    // Compatibility check: verify that old name is assigned to new 'output' node
    EXPECT_EQ(node_before_result_new->get_friendly_name(), node_name);
    // Compatibility check: Verify that old name is not set for old 'output' node anymore
    EXPECT_NE(node_before_result_old->get_friendly_name(), node_name);
}

TEST(pre_post_process, postprocess_keep_friendly_names_compatibility_implicit) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 10, 10});
    auto result_fr_name = f->get_results()[0]->get_friendly_name();
    auto node_before_result_old = f->get_results()[0]->get_input_source_output(0).get_node_shared_ptr();
    auto node_name = node_before_result_old->get_friendly_name();
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NCHW");
    p.output().tensor().set_layout("NHWC");
    f = p.build();
    EXPECT_EQ(f->get_results()[0]->get_friendly_name(), result_fr_name);
    auto node_before_result_new = f->get_results()[0]->get_input_source_output(0).get_node_shared_ptr();
    // Compatibility check: verify that old name is assigned to new 'output' node
    EXPECT_EQ(node_before_result_new->get_friendly_name(), node_name);
    // Compatibility check: Verify that old name is not set for old 'output' node anymore
    EXPECT_NE(node_before_result_old->get_friendly_name(), node_name);
}

// --- PostProcess - convert color format ---
TEST(pre_post_process, postprocess_convert_color_format_BGR_RGB) {
    auto f = create_simple_function(element::f32, Shape{5, 30, 20, 3});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::BGR);
    p.output().postprocess().convert_color(ColorFormat::RGB);
    f = p.build();

    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{5, 30, 20, 3}));
}

TEST(pre_post_process, postprocess_convert_color_format_RGB_BGR) {
    auto f = create_simple_function(element::f32, Shape{5, 30, 20, 3});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
    p.output().postprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{5, 30, 20, 3}));
}

TEST(pre_post_process, postprocess_convert_color_format_RGB_BGR_dynamic_batch) {
    auto f = create_simple_function(element::f32, PartialShape{-1, 30, 20, 3});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
    p.output().postprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{-1, 30, 20, 3}));
}

TEST(pre_post_process, postprocess_convert_color_format_RGB_BGR_dynamic_shape) {
    auto f = create_simple_function(element::f32, PartialShape{-1, -1, 20, 3});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
    p.output().postprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{-1, -1, 20, 3}));
}

TEST(pre_post_process, postprocess_convert_color_format_RGB_RGB) {
    auto f = create_simple_function(element::f32, Shape{5, 30, 20, 3});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
    p.output().postprocess().convert_color(ColorFormat::RGB);
    f = p.build();

    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{5, 30, 20, 3}));
}

TEST(pre_post_process, postprocess_convert_color_format_BGR_BGR) {
    auto f = create_simple_function(element::f32, Shape{5, 30, 20, 3});
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NHWC").set_color_format(ColorFormat::BGR);
    p.output().postprocess().convert_color(ColorFormat::BGR);
    f = p.build();

    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->get_result()->get_output_partial_shape(0), (PartialShape{5, 30, 20, 3}));
}

TEST(pre_post_process, postprocess_convert_color_format_unsupported) {
    auto f = create_simple_function(element::f32, Shape{5, 30, 20, 3});

    EXPECT_THROW(auto p = PrePostProcessor(f); p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
                 p.output().postprocess().convert_color(ColorFormat::GRAY);
                 f = p.build(), ov::Exception);

    EXPECT_THROW(auto p = PrePostProcessor(f); p.output().model().set_layout("NHWC").set_color_format(ColorFormat::RGB);
                 p.output().postprocess().convert_color(ColorFormat::UNDEFINED);
                 f = p.build(), ov::Exception);
    EXPECT_THROW(auto p = PrePostProcessor(f); p.output().model().set_color_format(ColorFormat::UNDEFINED);
                 p.output().postprocess().convert_color(ColorFormat::BGR);
                 f = p.build(), ov::AssertFailure);
}

// Postprocessing - other

TEST(pre_post_process, postprocess_preserve_rt_info) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    f->get_results()[0]->get_rt_info()["someKey"] = "someValue";
    f->input().get_rt_info()["someKey_in"] = "someValue_in";
    f->output().get_rt_info()["someKey_out"] = "someValue_out";
    auto p = PrePostProcessor(f);
    p.output().tensor().set_element_type(element::u8);
    f = p.build();
    EXPECT_EQ(f->output().get_element_type(), element::u8);

    ASSERT_EQ(f->get_results()[0]->get_rt_info().count("someKey"), 1);
    auto var0 = f->get_results()[0]->get_rt_info()["someKey"].as<std::string>();
    EXPECT_EQ(var0, "someValue");

    ASSERT_EQ(f->input().get_rt_info().count("someKey_in"), 1);
    auto var0_in = f->input().get_rt_info()["someKey_in"].as<std::string>();
    EXPECT_EQ(var0_in, "someValue_in");

    ASSERT_EQ(f->output().get_rt_info().count("someKey_out"), 1);
    auto var0_out = f->output().get_rt_info()["someKey_out"].as<std::string>();
    EXPECT_EQ(var0_out, "someValue_out");
}

TEST(pre_post_process, postprocess_custom_step) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    bool hit = false;
    auto p = PrePostProcessor(f);

    p.output().postprocess().custom([&hit](const ov::Output<Node>& node) {
        auto abs = std::make_shared<op::v0::Abs>(node);
        hit = true;
        return abs;
    });
    p.build();
    EXPECT_TRUE(hit);

    EXPECT_EQ(std::string(f->get_results()[0]->get_input_source_output(0).get_node()->get_type_name()),
              std::string(op::v0::Abs::get_type_info_static().name));
}

TEST(pre_post_process, postprocess_implicit_convert_element_type_and_layout) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);

    p.output().model().set_layout("NCHW");
    p.output().tensor().set_layout("NHWC").set_element_type(element::u8);
    p.build();
    EXPECT_EQ(f->get_results()[0]->get_element_type(), element::u8);
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->get_results()[0]->get_output_tensor(0).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
}

TEST(pre_post_process, postprocess_assert_output_without_index) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    EXPECT_ANY_THROW(p.output().tensor().set_element_type(element::f32); p.build());
    EXPECT_ANY_THROW(p.output("some_non_existing_name").tensor().set_element_type(element::f32); p.build());
}

TEST(pre_post_process, postprocess_keep_results_order) {
    auto f = create_n_inputs<3>(element::f32, Shape{1, 3, 2, 2});
    auto names0 = f->output(0).get_tensor().get_names();
    auto names1 = f->output(1).get_tensor().get_names();
    auto names2 = f->output(2).get_tensor().get_names();
    auto p = PrePostProcessor(f);

    p.output(0).model().set_layout("NCHW");
    p.output(1).model().set_layout("NCHW");
    p.output(1).tensor().set_layout("NHWC").set_element_type(element::u8);
    p.build();
    ASSERT_EQ(f->get_results().size(), 3);
    EXPECT_EQ(f->output(0).get_element_type(), element::f32);
    EXPECT_EQ(f->output(1).get_element_type(), element::u8);
    EXPECT_EQ(f->output(2).get_element_type(), element::f32);

    EXPECT_EQ(f->get_results()[0]->get_layout(), "NCHW") << f->get_results()[0]->get_layout().to_string();
    EXPECT_EQ(f->get_results()[1]->get_layout(), "NHWC") << f->get_results()[1]->get_layout().to_string();
    EXPECT_EQ(f->get_results()[2]->get_layout(), "") << f->get_results()[2]->get_layout().to_string();

    EXPECT_EQ(f->output(0).get_partial_shape(), (PartialShape{1, 3, 2, 2}));
    EXPECT_EQ(f->output(1).get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_EQ(f->output(2).get_partial_shape(), (PartialShape{1, 3, 2, 2}));

    EXPECT_EQ(f->output(0).get_tensor().get_names(), names0);
    EXPECT_EQ(f->output(1).get_tensor().get_names(), names1);
    EXPECT_EQ(f->output(2).get_tensor().get_names(), names2);
}

TEST(pre_post_process, postprocess_many) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    bool custom_called = false;

    auto p = PrePostProcessor(f);
    p.output("tensor_output1").model().set_layout("NCHW");
    p.output("tensor_output1")
        .postprocess()
        .convert_layout()
        .convert_element_type()
        .custom([&custom_called](const ov::Output<Node>& node) {
            custom_called = true;
            return std::make_shared<op::v0::Abs>(node);
        });
    p.output("tensor_output1").tensor().set_layout("NHWC").set_element_type(element::u8);

    f = p.build();
    EXPECT_EQ(f->get_results().size(), 1);
    EXPECT_EQ(f->output().get_tensor().get_names().count("tensor_output1"), 1);
    EXPECT_EQ(f->output().get_node_shared_ptr()->get_friendly_name(), "Result1");
    EXPECT_EQ(f->output().get_element_type(), element::u8);
    EXPECT_EQ(f->get_results()[0]->get_layout(), "NHWC");
    EXPECT_EQ(f->output().get_partial_shape(), (PartialShape{1, 2, 2, 3}));
    EXPECT_TRUE(custom_called);
}

TEST(pre_post_process, postprocess_one_node_many_outputs) {
    auto data1 = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto c1 = opset8::Constant::create(element::i32, Shape{}, {0});
    auto op = std::make_shared<opset8::Split>(data1, c1, 3);
    op->set_friendly_name("Split");
    ResultVector results;
    for (size_t i = 0; i < op->get_num_splits(); i++) {
        op->output(i).set_names({"tensor_Split" + std::to_string(i)});
        auto res = std::make_shared<op::v0::Result>(op->output(i));
        results.emplace_back(res);
    }
    auto model = std::make_shared<Model>(ResultVector{results}, ParameterVector{data1});
    EXPECT_EQ(model->output(0).get_tensor().get_names().count("tensor_Split0"), 1);
    EXPECT_EQ(model->output(1).get_tensor().get_names().count("tensor_Split1"), 1);
    EXPECT_EQ(model->output(2).get_tensor().get_names().count("tensor_Split2"), 1);

    auto p = PrePostProcessor(model);
    p.output(0).tensor().set_element_type(element::f32);
    p.output(2).tensor().set_element_type(element::f32);
    model = p.build();
    EXPECT_EQ(model->get_results().size(), 3);
    EXPECT_EQ(model->output(0).get_tensor().get_names().count("tensor_Split0"), 1);
    EXPECT_EQ(model->output(1).get_tensor().get_names().count("tensor_Split1"), 1);
    EXPECT_EQ(model->output(2).get_tensor().get_names().count("tensor_Split2"), 1);
    EXPECT_EQ(model->get_results()[0]->input(0).get_source_output().get_node()->get_friendly_name(), "Split.0");
    EXPECT_EQ(model->get_results()[1]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");
    EXPECT_EQ(model->get_results()[2]->input(0).get_source_output().get_node()->get_friendly_name(), "Split.2");
}

TEST(pre_post_process, postprocess_nothing_applied) {
    auto data1 = std::make_shared<op::v0::Parameter>(element::i32, Shape{1, 3, 10, 20});
    auto c1 = opset8::Constant::create(element::i32, Shape{}, {1});
    auto op = std::make_shared<opset8::Split>(data1, c1, 3);
    op->set_friendly_name("Split");
    ResultVector results;
    for (size_t i = 0; i < op->get_num_splits(); i++) {
        op->output(i).set_names({"tensor_Split" + std::to_string(i)});
        auto res = std::make_shared<op::v0::Result>(op->output(i));
        results.emplace_back(res);
    }
    auto model = std::make_shared<Model>(ResultVector{results}, ParameterVector{data1});
    EXPECT_EQ(model->get_results()[0]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");
    EXPECT_EQ(model->get_results()[1]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");
    EXPECT_EQ(model->get_results()[2]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");

    ov::layout::set_layout(model->output(1), "N???");
    auto p = PrePostProcessor(model);
    p.output(1).tensor().set_layout("NCHW");
    model = p.build();
    EXPECT_EQ(model->get_results().size(), 3);
    EXPECT_EQ(ov::layout::get_layout(model->output(1)), "NCHW");
    EXPECT_EQ(model->output(1).get_shape(), (Shape{1, 1, 10, 20}));
    EXPECT_EQ(model->output(0).get_tensor().get_names().count("tensor_Split0"), 1);
    EXPECT_EQ(model->output(1).get_tensor().get_names().count("tensor_Split1"), 1);
    EXPECT_EQ(model->output(2).get_tensor().get_names().count("tensor_Split2"), 1);
    EXPECT_EQ(model->get_results()[0]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");
    EXPECT_EQ(model->get_results()[1]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");
    EXPECT_EQ(model->get_results()[2]->input(0).get_source_output().get_node()->get_friendly_name(), "Split");
}

TEST(pre_post_process, exception_safety) {
    auto f = create_n_inputs<2>(element::f32, Shape{1, 3, 224, 224});
    auto name0 = f->input(0).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names0 = f->input(0).get_tensor().get_names();
    auto name1 = f->input(1).get_node_shared_ptr()->get_friendly_name();
    auto tensor_names1 = f->input(1).get_tensor().get_names();
    auto out_name0 = f->output(0).get_node_shared_ptr()->get_friendly_name();
    auto out_tensor_names0 = f->output(0).get_tensor().get_names();
    auto out_name1 = f->output(1).get_node_shared_ptr()->get_friendly_name();
    auto out_tensor_names1 = f->output(1).get_tensor().get_names();
    EXPECT_THROW(auto p = PrePostProcessor(f); p.input(0)  // this one is correct
                                                   .tensor()
                                                   .set_element_type(element::u8);
                 p.input(0).preprocess().convert_element_type(element::f32);
                 p.input(1)  // This one is not
                     .tensor()
                     .set_color_format(ColorFormat::NV12_TWO_PLANES);
                 p.input().preprocess().custom([](const Output<Node>& node) -> Output<Node> {
                     OPENVINO_THROW("test error");
                 });
                 p.build(), ov::AssertFailure);

    EXPECT_THROW(auto p = PrePostProcessor(f);

                 p.output(0)  // this one is correct
                     .tensor()
                     .set_element_type(element::u8);
                 p.output(1)  // This one is not
                     .postprocess()
                     .custom([](const Output<Node>& node) -> Output<Node> {
                         OPENVINO_THROW("test error");
                     });
                 p.build(), ov::Exception);
    EXPECT_EQ(f->get_parameters().size(), 2);

    EXPECT_EQ(f->input(0).get_element_type(), element::f32);
    EXPECT_EQ(f->input(0).get_partial_shape(), (PartialShape{1, 3, 224, 224}));
    EXPECT_EQ(f->input(0).get_node_shared_ptr()->get_friendly_name(), name0);
    EXPECT_EQ(f->input(0).get_tensor().get_names(), tensor_names0);

    EXPECT_EQ(f->input(1).get_element_type(), element::f32);
    EXPECT_EQ(f->input(1).get_partial_shape(), (PartialShape{1, 3, 224, 224}));
    EXPECT_EQ(f->input(1).get_node_shared_ptr()->get_friendly_name(), name1);
    EXPECT_EQ(f->input(1).get_tensor().get_names(), tensor_names1);

    EXPECT_EQ(f->output(0).get_node_shared_ptr()->get_friendly_name(), out_name0);
    EXPECT_EQ(f->output(0).get_tensor().get_names(), out_tensor_names0);

    EXPECT_EQ(f->output(1).get_node_shared_ptr()->get_friendly_name(), out_name1);
    EXPECT_EQ(f->output(1).get_tensor().get_names(), out_tensor_names1);
}

TEST(pre_post_process, layout_on_trivial) {
    auto f = create_trivial(element::f32, Shape{1, 440});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NC").set_element_type(element::f32);
    p.input().model().set_layout("NC");
    p.output().tensor().set_element_type(element::f32);
    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 440}));
    f = p.build();
    EXPECT_EQ(layout::get_layout(f->input()), "NC") << layout::get_layout(f->input()).to_string();
    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{1, 440}));
    ov::set_batch(f, 2);
    EXPECT_EQ(f->input().get_partial_shape(), (PartialShape{2, 440}));
}

TEST(pre_post_process, dump_empty) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    std::stringstream str;
    str << p;
    EXPECT_EQ(str.str(), std::string());
}

TEST(pre_post_process, dump_non_empty) {
    auto f = create_simple_function(element::f32, Shape{1, 3, 2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_memory_type("some_memory_type");
    std::stringstream str;
    str << p;
    EXPECT_NE(str.str(), std::string());
}

TEST(pre_post_process, dump_preprocess) {
    auto shape = PartialShape{1, 3, 2, 2};
    std::stringstream shape_stream;
    shape_stream << shape;
    auto shape_str = shape_stream.str();
    auto f = create_simple_function(element::f32, shape);
    auto p = PrePostProcessor(f);
    p.input()
        .tensor()
        .set_element_type(element::u8)
        .set_layout("NHWC")
        .set_spatial_dynamic_shape()
        .set_memory_type("test_memory_type");
    p.input()
        .preprocess()
        .convert_element_type(element::f32)
        .mean(1.f)
        .scale(2.f)
        .convert_layout({3, 2, 1, 0})
        .resize(ResizeAlgorithm::RESIZE_LINEAR, 480, 640)
        .resize(ResizeAlgorithm::RESIZE_LINEAR)
        .custom([](const Output<Node>& node) {
            return node;
        });
    p.input().model().set_layout("NCHW");
    std::stringstream stream;
    stream << p;
    auto dump = stream.str();
    EXPECT_TRUE(dump.find("Input") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("input1") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("memory type=test_memory_type") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Pre-processing steps (7):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("mean") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("scale") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert type") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert layout (3,2,1,0):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("resize to (480, 640):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("resize to model width/height:") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("custom:") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Implicit pre-processing steps (1):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert layout") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Model's expected tensor: " + shape_str + ", " + Layout("NCHW").to_string()) !=
                std::string::npos)
        << dump;
    EXPECT_TRUE(dump.find("output1") == std::string::npos) << dump;
}

TEST(pre_post_process, dump_preprocess_multiplane) {
    auto shape_to_string = [](const PartialShape& shape) {
        std::stringstream shape_stream;
        shape_stream << shape;
        return shape_stream.str();
    };
    auto shape = PartialShape{1, 3, 20, 20};
    auto shape_str = shape_to_string(shape);
    auto f = create_simple_function(element::f32, shape);
    auto p = PrePostProcessor(f);
    p.input().tensor().set_element_type(element::u8).set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_element_type(element::f32).convert_color(ColorFormat::RGB);
    p.input().model().set_layout("NCHW");
    std::stringstream stream;
    stream << p;
    auto dump = stream.str();
    EXPECT_TRUE(dump.find("Input") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("input1") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("memory type=") == std::string::npos) << dump;
    EXPECT_TRUE(dump.find("NV12") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("RGB") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Implicit pre-processing steps (1):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Pre-processing steps (2):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find(shape_to_string(PartialShape{1, 20, 20, 1})) != std::string::npos) << dump;
    EXPECT_TRUE(dump.find(shape_to_string(PartialShape{1, 10, 10, 2})) != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert type") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert color") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert layout") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Model's expected tensor: " + shape_str + ", " + Layout("NCHW").to_string()) !=
                std::string::npos)
        << dump;
    EXPECT_TRUE(dump.find("output1") == std::string::npos) << dump;
}

TEST(pre_post_process, dump_postprocess) {
    auto shape = PartialShape{1, 3, 2, 2};
    std::stringstream shape_stream;
    shape_stream << shape;
    auto shape_str = shape_stream.str();
    auto f = create_simple_function(element::f32, shape);
    auto p = PrePostProcessor(f);
    p.output().model().set_layout("NCHW");
    p.output()
        .postprocess()
        .convert_element_type(element::i32)
        .convert_layout({3, 2, 1, 0})
        .custom([](const Output<Node>& node) {
            return node;
        });
    p.output().tensor().set_element_type(element::u8).set_layout("NHWC");
    std::stringstream stream;
    stream << p;
    auto dump = stream.str();
    EXPECT_TRUE(dump.find("Output") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("output1") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Post-processing steps (3):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Post-processing implicit steps (2):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert type") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert layout (3,2,1,0):") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("convert layout " + Layout("NHWC").to_string()) != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("custom:") != std::string::npos) << dump;
    EXPECT_TRUE(dump.find("Model's data tensor: " + shape_str + ", " + Layout("NCHW").to_string()) != std::string::npos)
        << dump;
    EXPECT_TRUE(dump.find("input1") == std::string::npos) << dump;
}

TEST(pre_post_process, dump_error) {
    auto f = create_simple_function(element::f32, Shape{2, 2});
    auto p = PrePostProcessor(f);
    p.input().tensor().set_layout("NC");
    p.input().model().set_layout("HW");
    std::stringstream stream;
    stream << p;
    auto dump = stream.str();
    EXPECT_TRUE(dump.find("Error occurred:") != std::string::npos) << dump;
}
