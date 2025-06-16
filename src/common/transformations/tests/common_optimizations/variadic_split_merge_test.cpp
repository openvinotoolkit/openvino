// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/variadic_split_merge.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset3_decl.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

static ov::Output<ov::Node> make_strided_slice(const ov::Output<ov::Node>& out,
                                               const int64_t& start,
                                               const int64_t& stop,
                                               const int64_t& step,
                                               const int64_t& axis) {
    size_t num_dims = out.get_partial_shape().rank().is_static()
                          ? out.get_partial_shape().rank().get_length()
                          : throw std::runtime_error("Input rank must be static for strided slice creation");

    // Prepare default begin, end, strides for all axes
    std::vector<int64_t> begin(num_dims, 0);
    std::vector<int64_t> end(num_dims, 0);
    std::vector<int64_t> stride(num_dims, 1);

    // Set slicing axis values
    begin[axis] = start;
    end[axis] = stop;
    stride[axis] = step;

    std::vector<int64_t> mask(num_dims, 1);
    mask[axis] = 0;

    return std::make_shared<ov::op::v1::StridedSlice>(
        out,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{num_dims}, begin),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{num_dims}, end),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{num_dims}, stride),
        mask,  // begin mask
        mask   // end mask
    );
}

static ov::Output<ov::Node> make_slice(const ov::Output<ov::Node>& out,
                                       const int64_t& start,
                                       const int64_t& stop,
                                       const int64_t& step,
                                       const int64_t& axis) {
    return std::make_shared<ov::op::v8::Slice>(out,
                                               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {start}),
                                               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {stop}),
                                               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {step}),
                                               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis}));
}

static ov::OutputVector make_vsplit(const ov::Output<ov::Node>& out,
                                    const int64_t& axis,
                                    const std::vector<int64_t>& split_length) {
    return std::make_shared<ov::op::v1::VariadicSplit>(
               out,
               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis}),
               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{split_length.size()}, split_length))
        ->outputs();
}

TEST_F(TransformationTestsF, VariadicSplitMerge) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_slice(data, 32, 48, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeOutOfBounds) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_slice(data, 32, 100000, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeNegativeValue1) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_slice(data, 32, -1, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeNegativeValue2) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_slice(data, 16, -17, 1, 0);
        auto slice_2 = make_slice(data, -17, -1, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeReordered) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_slice(data, 32, 48, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_2, slice_1}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[2], vsplit[1]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergePartial) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 48, -1, -1});

        auto slice_0 = make_slice(data, 0, 16, 1, 1);
        auto slice_1 = make_slice(data, 16, 32, 1, 1);
        auto slice_2 = make_slice(data, 32, 48, 1, 1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 1);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 48, -1, -1});

        auto vsplit = make_vsplit(data, 1, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 1);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStrided1) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_strided_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_strided_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_strided_slice(data, 32, 48, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStrided2) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 48});

        auto slice_0 = make_strided_slice(data, 0, 16, 1, 1);
        auto slice_1 = make_strided_slice(data, 16, 32, 1, 1);
        auto slice_2 = make_strided_slice(data, 32, 48, 1, 1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 1);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 48});

        auto vsplit = make_vsplit(data, 1, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 1);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedOutOfBounds) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_strided_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_strided_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_strided_slice(data, 32, 5000, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedNegativeValue1) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_strided_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_strided_slice(data, 16, 32, 1, 0);
        auto slice_2 = make_strided_slice(data, 32, -1, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedNegativeValue2) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto slice_0 = make_strided_slice(data, 0, 16, 1, 0);
        auto slice_1 = make_strided_slice(data, 16, -17, 1, 0);
        auto slice_2 = make_strided_slice(data, -17, -1, 1, 0);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 0);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{48});

        auto vsplit = make_vsplit(data, 0, {16, 16, 16});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 0);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedPartial) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 9});

        auto slice_0 = make_strided_slice(data, 0, 3, 1, 2);
        auto slice_1 = make_strided_slice(data, 3, 6, 1, 2);
        auto slice_2 = make_strided_slice(data, 6, 9, 1, 2);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 2);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 9});

        auto vsplit = make_vsplit(data, 2, {3, 3, 3});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 2);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedHybridPartial) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 9});

        auto slice_0 = make_strided_slice(data, 0, 3, 1, 2);
        auto slice_1 = make_slice(data, 3, 6, 1, 2);
        auto slice_2 = make_strided_slice(data, 6, 9, 1, 2);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 2);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, 9});

        auto vsplit = make_vsplit(data, 2, {3, 3, 3});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 2);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedHybrid) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 9});

        auto slice_0 = make_strided_slice(data, 0, 3, 1, 2);
        auto slice_1 = make_slice(data, 3, 6, 1, 2);
        auto slice_2 = make_strided_slice(data, 6, 9, 1, 2);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 2);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 9});

        auto vsplit = make_vsplit(data, 2, {3, 3, 3});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 2);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, VariadicSplitMergeStridedMix) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 12});

        auto slice_0 = make_strided_slice(data, 0, 3, 1, 2);
        auto slice_1 = make_slice(data, 3, 6, 1, 2);
        auto slice_2 = make_strided_slice(data, 6, -1, 1, 2);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_1, slice_2}, 2);

        model = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::VariadicSplitMerge>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 9});

        auto vsplit = make_vsplit(data, 2, {3, 3, 6});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[1], vsplit[2]}, 2);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{concat}, ov::ParameterVector{data});
    }
}
