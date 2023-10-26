// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_strided_slice_to_crop.hpp>
#include <map>
#include <memory>
#include <ngraph/op/reshape.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertStridedSliceToCropTests1) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 384, 640});
        auto slice_begin = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 0, 0});
        auto slice_end = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 0, 0});
        auto slice_stride = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {1, 0, 1, 1};
        std::vector<int64_t> end_mask = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask = {0, 0, 0, 0};

        auto sslice = std::make_shared<ov::opset1::StridedSlice>(input,
                                                                 slice_begin,
                                                                 slice_end,
                                                                 slice_stride,
                                                                 begin_mask,
                                                                 end_mask,
                                                                 new_axis_mask,
                                                                 shrink_axis_mask,
                                                                 ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        model = std::make_shared<ov::Model>(ov::NodeVector{sslice}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 384, 640});

        std::vector<int64_t> axes = {0, 1, 2, 3};
        std::vector<int64_t> dim = {1, 1, 384, 640};
        std::vector<int64_t> offset = {0, 1, 0, 0};

        auto crop = std::make_shared<ngraph::op::CropIE>(input, axes, dim, offset);
        crop->set_friendly_name("strided_slice/Crop");

        auto reshape = ov::op::util::reshapeTo(crop, {1, 384, 640});
        reshape->set_friendly_name("strided_slice");

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reshape}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertStridedSliceToCropTests2) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 384, 640});
        auto slice_begin = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, 0});
        auto slice_end = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 0, 0});
        auto slice_stride = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {1, 1, 1, 1};
        std::vector<int64_t> end_mask = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask = {0, 0, 0, 0};

        auto sslice = std::make_shared<ov::opset1::StridedSlice>(input,
                                                                 slice_begin,
                                                                 slice_end,
                                                                 slice_stride,
                                                                 begin_mask,
                                                                 end_mask,
                                                                 new_axis_mask,
                                                                 shrink_axis_mask,
                                                                 ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        model = std::make_shared<ov::Model>(ov::NodeVector{sslice}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 384, 640});

        std::vector<int64_t> axes = {0, 1, 2, 3};
        std::vector<int64_t> dim = {1, 1, 384, 640};
        std::vector<int64_t> offset = {0, 1, 0, 0};

        auto crop = std::make_shared<ngraph::op::CropIE>(input, axes, dim, offset);
        crop->set_friendly_name("strided_slice/Crop");

        auto reshape = ov::op::util::reshapeTo(crop, {1, 384, 640});
        reshape->set_friendly_name("strided_slice");

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reshape}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertStridedSliceToCropNegative) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto slice_begin = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 0, 0});
        auto slice_end = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 0, 0});
        auto slice_stride = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {1, 0, 1, 1};
        std::vector<int64_t> end_mask = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask = {0, 0, 0, 0};

        auto sslice = std::make_shared<ov::opset1::StridedSlice>(input,
                                                                 slice_begin,
                                                                 slice_end,
                                                                 slice_stride,
                                                                 begin_mask,
                                                                 end_mask,
                                                                 new_axis_mask,
                                                                 shrink_axis_mask,
                                                                 ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        model = std::make_shared<ov::Model>(ov::NodeVector{sslice}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto slice_begin = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 0, 0});
        auto slice_end = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 0, 0});
        auto slice_stride = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {1, 0, 1, 1};
        std::vector<int64_t> end_mask = {1, 0, 1, 1};
        std::vector<int64_t> new_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> ellipsis_mask = {0, 0, 0, 0};

        auto sslice = std::make_shared<ov::opset1::StridedSlice>(input,
                                                                 slice_begin,
                                                                 slice_end,
                                                                 slice_stride,
                                                                 begin_mask,
                                                                 end_mask,
                                                                 new_axis_mask,
                                                                 shrink_axis_mask,
                                                                 ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{sslice}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertStridedSliceToCropNoneZeroBeginValuesWithMask) {
    // when begin_mask/end_mask are present begin/end values should not affect output shape
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});
        auto slice_begin = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 2, 1});
        auto slice_end = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, 2});
        auto slice_stride = ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {1, 0, 1, 1};
        std::vector<int64_t> end_mask = {1, 0, 1, 0};
        std::vector<int64_t> new_axis_mask = {0, 1, 0, 0};
        std::vector<int64_t> shrink_axis_mask = {0, 0, 0, 0};
        std::vector<int64_t> ellipsis_mask = {0, 0, 0, 0};

        auto sslice = std::make_shared<ov::opset1::StridedSlice>(input,
                                                                 slice_begin,
                                                                 slice_end,
                                                                 slice_stride,
                                                                 begin_mask,
                                                                 end_mask,
                                                                 new_axis_mask,
                                                                 shrink_axis_mask,
                                                                 ellipsis_mask);
        sslice->set_friendly_name("strided_slice");

        model = std::make_shared<ov::Model>(ov::NodeVector{sslice}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertStridedSliceToCropMatcher>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 4});

        std::vector<int64_t> axes = {0, 1, 2, 3};
        std::vector<int64_t> dim = {1, 1, 2, 2};
        std::vector<int64_t> offset = {0, 0, 0, 0};

        auto reshape = ov::op::util::reshapeTo(input, {1, 1, 2, 4});
        reshape->set_friendly_name("strided_slice/Reshape_for_Crop");

        auto crop = std::make_shared<ngraph::op::CropIE>(reshape, axes, dim, offset);
        crop->set_friendly_name("strided_slice");

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{crop}, ov::ParameterVector{input});
    }
}
