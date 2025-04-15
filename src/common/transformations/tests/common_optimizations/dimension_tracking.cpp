// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/dimension_tracking.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/detection_output.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/divide_fusion.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST(TransformationTests, AutoBatch_LabelPropagation_Transpose) {
    auto batch = ov::Dimension(5);
    auto A = std::make_shared<ov::Symbol>();

    batch.set_symbol(A);

    auto p_shape = ov::PartialShape{batch, 4, 6, 8};
    auto arg = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p_shape);
    auto input_order = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{2, 1, 0, 3});

    auto r = std::make_shared<ov::opset1::Transpose>(arg, input_order);

    EXPECT_EQ(r->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(r->get_output_partial_shape(0), ov::PartialShape({6, 4, batch, 8}));
    EXPECT_EQ(r->get_output_partial_shape(0)[2].get_symbol(), A);
}

TEST(TransformationTests, AutoBatch_LabelPropagation_Convolution) {
    auto batch = ov::Dimension(5);
    auto A = std::make_shared<ov::Symbol>();

    batch.set_symbol(A);

    auto p_shape = ov::PartialShape{batch, 4, 6, 8};
    auto arg = std::make_shared<ov::opset1::Parameter>(ov::element::f32, p_shape);

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 3, 3});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(arg,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    EXPECT_EQ(conv->get_output_element_type(0), ov::element::f32);
    EXPECT_EQ(conv->get_output_partial_shape(0), ov::PartialShape({batch, 1, 4, 6}));
    EXPECT_EQ(conv->get_output_partial_shape(0)[0].get_symbol(), A);
}

TEST(TransformationTests, AutoBatch_FindBatch_Transpose_and_Convolution) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{4, 1, 10, 10});

    const auto& order =
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    const auto& transpose = std::make_shared<ov::opset1::Transpose>(data, order);

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 3, 3});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(transpose,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(!shape[0].get_symbol()) << shape;
    ASSERT_TRUE(shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;

    const auto& out_shape = f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(out_shape[0].get_symbol()) << out_shape;
    ASSERT_TRUE(!out_shape[1].get_symbol()) << out_shape;
    ASSERT_TRUE(!out_shape[2].get_symbol()) << out_shape;
    ASSERT_TRUE(!out_shape[3].get_symbol()) << out_shape;
}

TEST(TransformationTests, AutoBatch_LabelPropagation_Convolution_Reshape) {
    auto data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4, 6, 8});

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 3, 3});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(data,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    const auto& reshape =
        std::make_shared<ov::opset1::Reshape>(conv,
                                              ov::opset1::Constant::create(ov::element::i64, {3}, {-1, 4, 6}),
                                              false);
    const auto& model = std::make_shared<ov::Model>(ov::NodeVector{reshape}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;

    const auto& out_shape = model->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(out_shape[0].get_symbol()) << out_shape;
    ASSERT_TRUE(!out_shape[1].get_symbol()) << out_shape;
    ASSERT_TRUE(!out_shape[2].get_symbol()) << out_shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_SingleMultiply) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});

    const auto& constant = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 1, 1});
    const auto& mul = std::make_shared<ov::opset1::Multiply>(data, constant);

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_Two_Outputs) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, 10, 10});

    const auto& order =
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    const auto& transpose = std::make_shared<ov::opset1::Transpose>(data, order);

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 1, 3, 3});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(data,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{conv, transpose}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_TwoOutputsReversed) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, 10, 10});

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 1, 3, 3});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(data,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    const auto& order =
        std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 0, 2, 3});
    const auto& transpose = std::make_shared<ov::opset1::Transpose>(data, order);

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{transpose, conv}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_IndependentBranchesConcated) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});

    const auto& constant_0 = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 1, 1, 1});
    const auto& mul_0 = std::make_shared<ov::opset1::Multiply>(data, constant_0);

    const auto& constant_1 = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 1, 1, 1});
    const auto& mul_1 = std::make_shared<ov::opset1::Multiply>(data, constant_1);

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 1, 1});
    const auto& conv = std::make_shared<ov::opset1::Convolution>(mul_0,
                                                                 filters,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});

    const auto& concat = std::make_shared<ov::opset1::Concat>(ov::NodeVector{conv, mul_1}, 1);

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_TwoConvNetwork) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 3, 3});
    const auto& conv_0 = std::make_shared<ov::opset1::Convolution>(data,
                                                                   filters,
                                                                   ov::Strides{1, 1},
                                                                   ov::CoordinateDiff{0, 0},
                                                                   ov::CoordinateDiff{0, 0},
                                                                   ov::Strides{1, 1});

    const auto& conv_1 = std::make_shared<ov::opset1::Convolution>(data,
                                                                   filters,
                                                                   ov::Strides{1, 1},
                                                                   ov::CoordinateDiff{0, 0},
                                                                   ov::CoordinateDiff{0, 0},
                                                                   ov::Strides{1, 1});

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{conv_0, conv_1}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>();
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_NegativeTracking) {
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4, 10, 10});

    const auto& filters = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1, 4, 3, 3});
    const auto& conv_0 = std::make_shared<ov::opset1::Convolution>(data,
                                                                   filters,
                                                                   ov::Strides{1, 1},
                                                                   ov::CoordinateDiff{0, 0},
                                                                   ov::CoordinateDiff{0, 0},
                                                                   ov::Strides{1, 1});
    const auto& pattern = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
    const auto& reshape = std::make_shared<ov::opset1::Reshape>(conv_0, pattern, false);

    const auto& f = std::make_shared<ov::Model>(ov::NodeVector{reshape}, ov::ParameterVector{data});

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>(false, false);
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;

    const auto& out_shape = f->get_results()[0]->get_output_partial_shape(0);
    ASSERT_TRUE(!out_shape[0].get_symbol()) << out_shape;
}

TEST(TransformationTests, AutoBatch_FindBatch_AutoBatch_LabelPropagation_DO_detachment) {
    auto f = ov::test::utils::make_detection_output();
    auto& data = f->get_parameters()[0];

    ov::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::pass::FindBatch>(true);
    m.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    const auto& shape = data->get_partial_shape();
    ASSERT_TRUE(shape[0].get_symbol()) << shape;
    ASSERT_TRUE(!shape[1].get_symbol()) << shape;
    ASSERT_TRUE(!shape[2].get_symbol()) << shape;
    ASSERT_TRUE(!shape[3].get_symbol()) << shape;
    ASSERT_EQ(f->get_results().size(), 3);
    for (const auto& result : f->get_results()) {
        const auto& out_shape = result->get_output_partial_shape(0);
        ASSERT_TRUE(out_shape[0].get_symbol()) << out_shape;
        ASSERT_TRUE(!out_shape[1].get_symbol()) << out_shape;
    }
}

TEST(partial_shape, cout_with_label) {
    ov::Dimension a = 5;
    auto A = std::make_shared<ov::Symbol>();
    a.set_symbol(A);
    ov::PartialShape shape{1, 2, 3, a};
    std::stringstream stream;
    stream << shape;
    ASSERT_EQ(stream.str(), "[1,2,3,5]");
}

TEST(partial_shape, cout_without_label) {
    ov::Dimension a = 5;
    ov::PartialShape shape{1, 2, 3, a};
    std::stringstream stream;
    stream << shape;
    ASSERT_EQ(stream.str(), "[1,2,3,5]");
}
