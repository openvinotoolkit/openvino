// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

TEST(TransformationTests, ConstFoldingPriorBox) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    {
        auto in = std::make_shared<opset3::Parameter>(element::i64, Shape{2});
        op::v0::PriorBox::Attributes attrs;
        attrs.min_size = {256.0f};
        attrs.max_size = {315.0f};
        attrs.aspect_ratio = {2.0f};
        attrs.flip = true;
        attrs.scale_all_sizes = true;

        auto layer_shape = opset3::Constant::create<int64_t>(element::i64, Shape{2}, {1, 1});
        auto image_shape = opset3::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
        auto pb = std::make_shared<opset3::PriorBox>(layer_shape, image_shape, attrs);
        auto res = std::make_shared<opset3::Result>(pb);
        f = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{in});
        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto layer_shape = std::make_shared<opset3::Parameter>(element::i64, Shape{2});
        auto const_prior_box = opset3::Constant::create<float>(
            element::f32,
            Shape{2, 16},
            {
                -0.426667f, -0.426667f, 0.426667f, 0.426667f, -0.473286f, -0.473286f, 0.473286f, 0.473286f,
                -0.603398f, -0.301699f, 0.603398f, 0.301699f, -0.301699f, -0.603398f, 0.301699f, 0.603398f,
                0.1f,       0.1f,       0.1f,      0.1f,      0.1f,       0.1f,       0.1f,      0.1f,
                0.1f,       0.1f,       0.1f,      0.1f,      0.1f,       0.1f,       0.1f,      0.1f,
            });
        auto res = std::make_shared<opset3::Result>(const_prior_box);
        f_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{layer_shape});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto fused = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());
    auto ref = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());

    EXPECT_TRUE(fused != nullptr);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(fused->get_vector<float>() == ref->get_vector<float>());
}

TEST(TransformationTests, ConstFoldingPriorBoxClustered) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    {
        auto in = std::make_shared<opset3::Parameter>(element::i64, Shape{2});
        op::v0::PriorBoxClustered::Attributes attrs;
        attrs.widths = {4.0f, 2.0f, 3.2f};
        attrs.heights = {1.0f, 2.0f, 1.1f};

        auto layer_shape = opset3::Constant::create<int64_t>(element::i64, Shape{2}, {2, 2});
        auto image_shape = opset3::Constant::create<int64_t>(element::i64, Shape{2}, {300, 300});
        auto pb = std::make_shared<opset3::PriorBoxClustered>(layer_shape, image_shape, attrs);
        auto res = std::make_shared<opset3::Result>(pb);
        f = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{in});
        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto layer_shape = std::make_shared<opset3::Parameter>(element::i64, Shape{2});
        auto const_prior_box = opset3::Constant::create<float>(
            element::f32,
            Shape{2, 48},
            {-0.00666667f, -0.00166667f, 0.00666667f, 0.00166667f, -0.00333333f, -0.00333333f, 0.00333333f, 0.00333333f,
             -0.00533333f, -0.00183333f, 0.00533333f, 0.00183333f, -0.00333333f, -0.00166667f, 0.01f,       0.00166667f,
             0.0f,         -0.00333333f, 0.00666667f, 0.00333333f, -0.002f,      -0.00183333f, 0.00866667f, 0.00183333f,
             -0.00666667f, 0.00166667f,  0.00666667f, 0.005f,      -0.00333333f, 0.0f,         0.00333333f, 0.00666667f,
             -0.00533333f, 0.0015f,      0.00533333f, 0.00516667f, -0.00333333f, 0.00166667f,  0.01f,       0.005f,
             0.0f,         0.0f,         0.00666667f, 0.00666667f, -0.002f,      0.0015f,      0.00866667f, 0.00516667f,
             0.1f,         0.1f,         0.1f,        0.1f,        0.1f,         0.1f,         0.1f,        0.1f,
             0.1f,         0.1f,         0.1f,        0.1f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f});
        auto res = std::make_shared<opset3::Result>(const_prior_box);
        f_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{layer_shape});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto fused = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());
    auto ref = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());

    EXPECT_TRUE(fused != nullptr);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(fused->get_vector<float>() == ref->get_vector<float>());
}

TEST(TransformationTests, ConstFoldingPriorBoxSubgraph) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    {
        auto in = std::make_shared<opset3::Parameter>(element::i64, Shape{2, 3, 1, 1});
        auto in_2 = std::make_shared<opset3::Parameter>(element::i64, Shape{2, 3, 300, 300});
        op::v0::PriorBox::Attributes attrs;
        attrs.min_size = {256.0f};
        attrs.max_size = {315.0f};
        attrs.aspect_ratio = {2.0f};
        attrs.flip = true;
        attrs.scale_all_sizes = true;

        auto layer_shape = std::make_shared<opset3::ShapeOf>(in);
        auto image_shape = std::make_shared<opset3::ShapeOf>(in_2);

        auto begin = opset3::Constant::create(element::i64, Shape{1}, {2});
        auto end = opset3::Constant::create(element::i64, Shape{1}, {4});
        auto stride = opset3::Constant::create(element::i64, Shape{1}, {1});
        auto ss_data = std::make_shared<opset3::StridedSlice>(layer_shape,
                                                              begin,
                                                              end,
                                                              stride,
                                                              std::vector<int64_t>{0},
                                                              std::vector<int64_t>{0});

        auto ss_image = std::make_shared<opset3::StridedSlice>(image_shape,
                                                               begin,
                                                               end,
                                                               stride,
                                                               std::vector<int64_t>{0},
                                                               std::vector<int64_t>{0});
        auto pb = std::make_shared<opset3::PriorBox>(ss_data, ss_image, attrs);
        auto res = std::make_shared<opset3::Result>(pb);
        f = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{in, in_2});
        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto layer_shape = std::make_shared<opset3::Parameter>(element::i64, Shape{2});
        auto const_prior_box = opset3::Constant::create<float>(
            element::f32,
            Shape{2, 16},
            {-0.426667f, -0.426667f, 0.426667f, 0.426667f, -0.473286f, -0.473286f, 0.473286f, 0.473286f,
             -0.603398f, -0.301699f, 0.603398f, 0.301699f, -0.301699f, -0.603398f, 0.301699f, 0.603398f,
             0.1f,       0.1f,       0.1f,      0.1f,      0.1f,       0.1f,       0.1f,      0.1f,
             0.1f,       0.1f,       0.1f,      0.1f,      0.1f,       0.1f,       0.1f,      0.1f});
        auto res = std::make_shared<opset3::Result>(const_prior_box);
        f_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{layer_shape});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto fused = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());
    auto ref = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());

    EXPECT_TRUE(fused != nullptr);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(fused->get_vector<float>() == ref->get_vector<float>());
}

TEST(TransformationTests, ConstFoldingPriorBoxClusteredSubgraph) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto in = std::make_shared<opset3::Parameter>(element::i64, Shape{2, 3, 2, 2});
        auto in_2 = std::make_shared<opset3::Parameter>(element::i64, Shape{2, 3, 300, 300});
        op::v0::PriorBoxClustered::Attributes attrs;
        attrs.widths = {4.0f, 2.0f, 3.2f};
        attrs.heights = {1.0f, 2.0f, 1.1f};

        auto layer_shape = std::make_shared<opset3::ShapeOf>(in);
        auto image_shape = std::make_shared<opset3::ShapeOf>(in_2);

        auto begin = opset3::Constant::create(element::i64, Shape{1}, {2});
        auto end = opset3::Constant::create(element::i64, Shape{1}, {4});
        auto stride = opset3::Constant::create(element::i64, Shape{1}, {1});
        auto ss_data = std::make_shared<opset3::StridedSlice>(layer_shape,
                                                              begin,
                                                              end,
                                                              stride,
                                                              std::vector<int64_t>{0},
                                                              std::vector<int64_t>{0});

        auto ss_image = std::make_shared<opset3::StridedSlice>(image_shape,
                                                               begin,
                                                               end,
                                                               stride,
                                                               std::vector<int64_t>{0},
                                                               std::vector<int64_t>{0});
        auto pb = std::make_shared<opset3::PriorBoxClustered>(ss_data, ss_image, attrs);
        auto res = std::make_shared<opset3::Result>(pb);
        f = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{in, in_2});
        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto layer_shape = std::make_shared<opset3::Parameter>(element::i64, Shape{2});
        auto const_prior_box = opset3::Constant::create<float>(
            element::f32,
            Shape{2, 48},
            {-0.00666667f, -0.00166667f, 0.00666667f, 0.00166667f, -0.00333333f, -0.00333333f, 0.00333333f, 0.00333333f,
             -0.00533333f, -0.00183333f, 0.00533333f, 0.00183333f, -0.00333333f, -0.00166667f, 0.01f,       0.00166667f,
             0.0f,         -0.00333333f, 0.00666667f, 0.00333333f, -0.002f,      -0.00183333f, 0.00866667f, 0.00183333f,
             -0.00666667f, 0.00166667f,  0.00666667f, 0.005f,      -0.00333333f, 0.0f,         0.00333333f, 0.00666667f,
             -0.00533333f, 0.0015f,      0.00533333f, 0.00516667f, -0.00333333f, 0.00166667f,  0.01f,       0.005f,
             0.0f,         0.0f,         0.00666667f, 0.00666667f, -0.002f,      0.0015f,      0.00866667f, 0.00516667f,
             0.1f,         0.1f,         0.1f,        0.1f,        0.1f,         0.1f,         0.1f,        0.1f,
             0.1f,         0.1f,         0.1f,        0.1f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f,
             0.0f,         0.0f,         0.0f,        0.0f,        0.0f,         0.0f,         0.0f,        0.0f});
        auto res = std::make_shared<opset3::Result>(const_prior_box);
        f_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{layer_shape});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto fused = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());
    auto ref = ov::as_type_ptr<opset3::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());

    EXPECT_TRUE(fused != nullptr);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(fused->get_vector<float>() == ref->get_vector<float>());
}

TEST(TransformationTests, ConstFoldingPriorBox8) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    {
        auto in = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        op::v8::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;
        attrs.min_max_aspect_ratios_order = false;

        auto layer_shape = opset8::Constant::create<int64_t>(element::i64, Shape{2}, {2, 2});
        auto image_shape = opset8::Constant::create<int64_t>(element::i64, Shape{2}, {10, 10});
        auto pb = std::make_shared<opset8::PriorBox>(layer_shape, image_shape, attrs);
        auto res = std::make_shared<opset8::Result>(pb);
        f = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{in});
        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto layer_shape = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto const_prior_box = opset8::Constant::create<float>(
            element::f32,
            Shape{2, 48},
            {0.15f,      0.15f,     0.35f,     0.35f,      0.127526f, 0.16835f,   0.372474f, 0.33165f,  0.0918861f,
             0.0918861f, 0.408114f, 0.408114f, 0.65f,      0.15f,     0.85f,      0.35f,     0.627526f, 0.16835f,
             0.872474f,  0.33165f,  0.591886f, 0.0918861f, 0.908114f, 0.408114f,  0.15f,     0.65f,     0.35f,
             0.85f,      0.127526f, 0.66835f,  0.372474f,  0.83165f,  0.0918861f, 0.591886f, 0.408114f, 0.908114f,
             0.65f,      0.65f,     0.85f,     0.85f,      0.627526f, 0.66835f,   0.872474f, 0.83165f,  0.591886f,
             0.591886f,  0.908114f, 0.908114f, 0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f});
        auto res = std::make_shared<opset8::Result>(const_prior_box);
        f_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{layer_shape});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto fused = ov::as_type_ptr<opset8::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());
    auto ref = ov::as_type_ptr<opset8::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());

    EXPECT_TRUE(fused != nullptr);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(fused->get_vector<float>() == ref->get_vector<float>());
}

TEST(TransformationTests, ConstFoldingPriorBox8Subgraph) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);

    {
        auto in = std::make_shared<opset8::Parameter>(element::i64, Shape{2, 3, 2, 2});
        auto in_2 = std::make_shared<opset8::Parameter>(element::i64, Shape{2, 3, 10, 10});
        op::v8::PriorBox::Attributes attrs;
        attrs.min_size = {2.0f};
        attrs.max_size = {5.0f};
        attrs.aspect_ratio = {1.5f};
        attrs.scale_all_sizes = true;
        attrs.min_max_aspect_ratios_order = false;

        auto layer_shape = std::make_shared<opset8::ShapeOf>(in);
        auto image_shape = std::make_shared<opset8::ShapeOf>(in_2);

        auto begin = opset8::Constant::create(element::i64, Shape{1}, {2});
        auto end = opset8::Constant::create(element::i64, Shape{1}, {4});
        auto stride = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto ss_data = std::make_shared<opset8::StridedSlice>(layer_shape,
                                                              begin,
                                                              end,
                                                              stride,
                                                              std::vector<int64_t>{0},
                                                              std::vector<int64_t>{0});

        auto ss_image = std::make_shared<opset8::StridedSlice>(image_shape,
                                                               begin,
                                                               end,
                                                               stride,
                                                               std::vector<int64_t>{0},
                                                               std::vector<int64_t>{0});
        auto pb = std::make_shared<opset8::PriorBox>(ss_data, ss_image, attrs);
        auto res = std::make_shared<opset8::Result>(pb);
        f = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{in, in_2});
        pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto layer_shape = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto const_prior_box = opset8::Constant::create<float>(
            element::f32,
            Shape{2, 48},
            {0.15f,      0.15f,     0.35f,     0.35f,      0.127526f, 0.16835f,   0.372474f, 0.33165f,  0.0918861f,
             0.0918861f, 0.408114f, 0.408114f, 0.65f,      0.15f,     0.85f,      0.35f,     0.627526f, 0.16835f,
             0.872474f,  0.33165f,  0.591886f, 0.0918861f, 0.908114f, 0.408114f,  0.15f,     0.65f,     0.35f,
             0.85f,      0.127526f, 0.66835f,  0.372474f,  0.83165f,  0.0918861f, 0.591886f, 0.408114f, 0.908114f,
             0.65f,      0.65f,     0.85f,     0.85f,      0.627526f, 0.66835f,   0.872474f, 0.83165f,  0.591886f,
             0.591886f,  0.908114f, 0.908114f, 0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f,       0.1f,      0.1f,      0.1f,
             0.1f,       0.1f,      0.1f,      0.1f,       0.1f,      0.1f});
        auto res = std::make_shared<opset8::Result>(const_prior_box);
        f_ref = std::make_shared<ov::Model>(NodeVector{res}, ParameterVector{layer_shape});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto fused = ov::as_type_ptr<opset8::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());
    auto ref = ov::as_type_ptr<opset8::Constant>(f->get_result()->input_value(0).get_node_shared_ptr());

    EXPECT_TRUE(fused != nullptr);
    EXPECT_TRUE(ref != nullptr);
    EXPECT_TRUE(fused->get_vector<float>() == ref->get_vector<float>());
}
