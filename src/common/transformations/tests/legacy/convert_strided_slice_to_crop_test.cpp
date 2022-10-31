// Copyright (C) 2018-2022 Intel Corporation
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

TEST_F(TransformationTestsF, ConvertStridedSliceToCropTests1) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertStridedSliceToCropTests2) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertStridedSliceToCropNegative) {
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertStridedSliceToCropNoneZeroBeginValuesWithMask) {
    // when begin_mask/end_mask are present begin/end values should not affect output shape
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

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sslice}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
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

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{crop}, ngraph::ParameterVector{input});
    }
}
