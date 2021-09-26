// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>
#include <vector>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <legacy/ngraph_ops/crop_ie.hpp>

#include <legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>
#include <ngraph/op/reshape.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertStridedSliceToCropTests1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 384, 640});
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 0, 0});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 0, 0});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask       = {1, 0, 1, 1};
        std::vector<int64_t> end_mask         = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask    = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 384, 640});

        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 384, 640};
        std::vector<int64_t> offset = {0, 1, 0, 0};

        auto crop         = std::make_shared<ngraph::op::CropIE>(input, axes, dim, offset);
        crop->set_friendly_name("strided_slice/Crop");

        auto reshape      = ngraph::op::util::reshapeTo(crop, {1, 384, 640});
        reshape->set_friendly_name("strided_slice");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto reshape_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = reshape_node->input(0).get_source_output().get_node_shared_ptr();
    bool names_are_correct = (crop_node->get_friendly_name() == "strided_slice/Crop") &&
                             (reshape_node->get_friendly_name() == "strided_slice");
    ASSERT_TRUE(names_are_correct) << "Transformation ConvertStridedSliceToCrop should keep output names.\n";
}

TEST(TransformationTests, ConvertStridedSliceToCropTests2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 384, 640});
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 0});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 0, 0});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask       = {1, 1, 1, 1};
        std::vector<int64_t> end_mask         = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask    = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 384, 640});

        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 384, 640};
        std::vector<int64_t> offset = {0, 1, 0, 0};

        auto crop         = std::make_shared<ngraph::op::CropIE>(input, axes, dim, offset);
        crop->set_friendly_name("strided_slice/Crop");

        auto reshape      = ngraph::op::util::reshapeTo(crop, {1, 384, 640});
        reshape->set_friendly_name("strided_slice");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto reshape_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = reshape_node->input(0).get_source_output().get_node_shared_ptr();
    bool names_are_correct = (crop_node->get_friendly_name() == "strided_slice/Crop") &&
                             (reshape_node->get_friendly_name() == "strided_slice");
    ASSERT_TRUE(names_are_correct) << "Transformation ConvertStridedSliceToCrop should keep output names.\n";
}

TEST(TransformationTests, ConvertStridedSliceToCropNegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 0, 0});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 0, 0});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask       = {1, 0, 1, 1};
        std::vector<int64_t> end_mask         = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask    = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 1, 0, 0});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 0, 0});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask       = {1, 0, 1, 1};
        std::vector<int64_t> end_mask         = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask    = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

// in this test the Crop will get 3D input which is not supported so the transformation will not be applied
TEST(TransformationTests, ConvertStridedSliceToCropNegative2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{128, 1});
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 0, 0});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 0, 0});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 1});

        std::vector<int64_t> begin_mask       = {0, 1, 1};
        std::vector<int64_t> end_mask         = {0, 1, 1};
        std::vector<int64_t> new_axis_mask    = {1, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{128, 1});
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 0, 0});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 0, 0});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 1});

        std::vector<int64_t> begin_mask       = {0, 1, 1};
        std::vector<int64_t> end_mask         = {0, 1, 1};
        std::vector<int64_t> new_axis_mask    = {1, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}


TEST(TransformationTests, ConvertStridedSliceToCropNoneZeroBeginValuesWithMask) {
    // when begin_mask/end_mask are present begin/end values should not affect output shape
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input        = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 4});
        auto slice_begin  = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 2, 1});
        auto slice_end    = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 0, 0, 2});
        auto slice_stride = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask       = {1, 0, 1, 1};
        std::vector<int64_t> end_mask         = {1, 0, 1, 0};
        std::vector<int64_t> new_axis_mask    = {0, 1, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> ellipsis_mask    = {0, 0, 0, 0};

        auto sslice = std::make_shared<ngraph::opset1::StridedSlice>(input, slice_begin, slice_end, slice_stride,
                                                                     begin_mask, end_mask,
                                                                     new_axis_mask, shrink_axis_mask, ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 4});

        std::vector<int64_t> axes   = {0, 1, 2, 3};
        std::vector<int64_t> dim    = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};

        auto reshape = ngraph::op::util::reshapeTo(input, {1, 1, 2, 4});
        reshape->set_friendly_name("strided_slice/Reshape_for_Crop");

        auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);
        crop->set_friendly_name("strided_slice");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{crop}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
