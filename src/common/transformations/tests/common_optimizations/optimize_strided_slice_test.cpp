// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/optimize_strided_slice.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/op_conversions/convert_slice_to_strided_slice.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion_Negative1) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};  // ignoring end -- slicing to the end

        auto ss = std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
}

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion_Negative2) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(4));
        auto relu = std::make_shared<opset1::Relu>(data);
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};  // ignoring end -- slicing to the end

        auto ss = std::make_shared<opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask);

        model = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
}

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto relu = std::make_shared<opset1::Relu>(data);
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};  // ignoring end -- slicing to the end

        auto ss = std::make_shared<opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask);

        model = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto relu = std::make_shared<opset1::Relu>(data);
        model_ref = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_SkipUselessDeletionRevertCase) {
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto begin = opset3::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset3::Constant::create(element::i64, Shape{4}, {-6, -7, -8, -9});
        auto stride = opset3::Constant::create(element::i64, Shape{4}, {-1});

        std::vector<int64_t> begin_mask = {1, 1, 1, 1};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<opset3::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);
        auto relu = std::make_shared<opset3::Relu>(ss);

        model = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<opset3::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto begin = opset3::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset3::Constant::create(element::i64, Shape{4}, {-6, -7, -8, -9});
        auto stride = opset3::Constant::create(element::i64, Shape{4}, {-1});

        std::vector<int64_t> begin_mask = {1, 1, 1, 1};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<opset3::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);
        auto relu = std::make_shared<opset3::Relu>(ss);

        model_ref = std::make_shared<ov::Model>(NodeVector{relu}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, UselessSlice) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto relu = std::make_shared<opset1::Relu>(data);
        auto begin = opset1::Constant::create(element::i64, Shape{2}, {0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{2}, {5, 5});
        auto stride = opset1::Constant::create(element::i64, Shape{2}, {1, 1});
        auto axis = opset1::Constant::create(element::i64, Shape{2}, {1, 3});

        auto slice = std::make_shared<opset8::Slice>(relu, begin, end, stride, axis);

        model = std::make_shared<ov::Model>(slice, ParameterVector{data});
        manager.register_pass<ov::pass::UselessSliceEraser>();
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto relu = std::make_shared<opset1::Relu>(data);
        model_ref = std::make_shared<ov::Model>(relu, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, NegativeUselessSliceWithNegativeStrides) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto relu = std::make_shared<opset1::Relu>(data);
        auto begin = opset1::Constant::create(element::i64, Shape{2}, {4, 0});
        auto end = opset1::Constant::create(element::i64, Shape{2}, {INT32_MIN, 5});
        auto stride = opset1::Constant::create(element::i64, Shape{2}, {-1, 1});
        auto axis = opset1::Constant::create(element::i64, Shape{2}, {1, 3});

        auto slice = std::make_shared<opset8::Slice>(relu, begin, end, stride, axis);

        model = std::make_shared<ov::Model>(slice, ParameterVector{data});
        manager.register_pass<ov::pass::UselessSliceEraser>();
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Usefull_Test) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto ss = std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Shared_Test) {
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end2 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride2 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<opset1::StridedSlice>(source, begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss2}, 0);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss1}, 0);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_NotShared_Test) {
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 6, 5, 5});

        auto axis = opset1::Constant::create(element::i64, Shape{}, {1});
        auto split = std::make_shared<opset1::Split>(source, axis, 2);

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 =
            std::make_shared<opset1::StridedSlice>(split->output(0), begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end2 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride2 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 =
            std::make_shared<opset1::StridedSlice>(split->output(1), begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss2}, 0);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 6, 5, 5});

        auto axis = opset1::Constant::create(element::i64, Shape{}, {1});
        auto split = std::make_shared<opset1::Split>(source, axis, 2);

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 =
            std::make_shared<opset1::StridedSlice>(split->output(0), begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end2 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride2 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 =
            std::make_shared<opset1::StridedSlice>(split->output(1), begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss2}, 0);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Groupped_Test) {
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {5, 3, 5, 5});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = opset1::Constant::create(element::i64, Shape{4}, {0, 3, 0, 0});
        auto end2 = opset1::Constant::create(element::i64, Shape{4}, {5, 5, 5, 5});
        auto stride2 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<opset1::StridedSlice>(source, begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss2}, 1);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto axis = opset1::Constant::create(element::i64, Shape{}, {1});
        auto split_sizes = opset1::Constant::create(element::i64, Shape{2}, {3, 2});
        auto variadic_split = std::make_shared<opset1::VariadicSplit>(source, axis, split_sizes);

        auto concat =
            std::make_shared<opset1::Concat>(OutputVector{variadic_split->output(0), variadic_split->output(1)}, 1);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_UselessDeletion_use_shapes_false) {
    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});
        auto relu = std::make_shared<opset1::Relu>(data);
        auto begin = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset1::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {1, 1, 1, 1};  // ignoring end -- slicing to the end

        auto ss = std::make_shared<opset1::StridedSlice>(relu, begin, end, stride, begin_mask, end_mask);

        model = std::make_shared<ov::Model>(NodeVector{ss}, ParameterVector{data});
        manager.register_pass<ov::pass::StridedSliceOptimization>(false);
        manager.register_pass<pass::ConstantFolding>();
    }
    // No UselessStridedSliceEraser transformation if use_shapes == false
}

TEST_F(TransformationTestsF, OptimizeSS_Shared_Test_use_shapes_false) {
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end2 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride2 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<opset1::StridedSlice>(source, begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss2}, 0);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
        manager.register_pass<ov::pass::StridedSliceOptimization>(false);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss1}, 0);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
    }
}

TEST_F(TransformationTestsF, OptimizeSS_Groupped_Test_use_shapes_false) {
    {
        auto source = std::make_shared<opset1::Parameter>(element::f32, Shape{5, 5, 5, 5});

        auto begin1 = opset1::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end1 = opset1::Constant::create(element::i64, Shape{4}, {5, 3, 5, 5});
        auto stride1 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask1 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask1 = {0, 0, 0, 0};
        auto ss1 = std::make_shared<opset1::StridedSlice>(source, begin1, end1, stride1, begin_mask1, end_mask1);

        auto begin2 = opset1::Constant::create(element::i64, Shape{4}, {0, 3, 0, 0});
        auto end2 = opset1::Constant::create(element::i64, Shape{4}, {5, 5, 5, 5});
        auto stride2 = opset1::Constant::create(element::i64, Shape{4}, {1});
        std::vector<int64_t> begin_mask2 = {0, 0, 0, 0};
        std::vector<int64_t> end_mask2 = {0, 0, 0, 0};
        auto ss2 = std::make_shared<opset1::StridedSlice>(source, begin2, end2, stride2, begin_mask2, end_mask2);

        auto concat = std::make_shared<opset1::Concat>(NodeVector{ss1, ss2}, 1);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{source});
        manager.register_pass<ov::pass::StridedSliceOptimization>(false);
        manager.register_pass<pass::ConstantFolding>();
    }
    // No GroupedStridedSliceOptimizer transformation if use_shapes == false
}

TEST_F(TransformationTestsF, SliceToStridedSlice_default_axes) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto step = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<opset8::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_axes_const_sorted_full) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto step = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

        auto axes = opset8::Constant::create(element::i64, Shape{4}, {0, 1, 2, 3});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto stride = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_all_const) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{4}, {2, 3, 4, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto end = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto step = opset8::Constant::create(element::i64, Shape{1}, {1});

        auto axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = opset8::Constant::create(element::f32, Shape{4}, {2, 3, 4, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto end = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto stride = opset8::Constant::create(element::i64, Shape{1}, {1});

        std::vector<int64_t> begin_end_mask = {0};
        auto strided_slice =
            std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_all_const_fold) {
    {
        auto data = opset8::Constant::create(element::f32, Shape{4}, {2, 3, 4, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto end = opset8::Constant::create(element::i64, Shape{1}, {-1});
        auto step = opset8::Constant::create(element::i64, Shape{1}, {1});

        auto axes = opset8::Constant::create(element::i64, Shape{1}, {-1});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto sliced_const = opset8::Constant::create(element::f32, Shape{2}, {3, 4});
        model_ref = std::make_shared<ov::Model>(NodeVector{sliced_const}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_axes_const_sorted_less) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{2});

        auto axes = opset8::Constant::create(element::i64, Shape{2}, {1, 2});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto start = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto stop = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{2});

        auto axes = opset8::Constant::create(element::i64, Shape{2}, {1, 2});
        auto zero = opset8::Constant::create(element::i32, Shape{1}, {0});

        const auto default_begin = opset8::Constant::create(element::i64, Shape{3}, {0});
        const auto begin = std::make_shared<opset8::ScatterUpdate>(default_begin, axes, start, zero);

        const auto default_end = opset8::Constant::create(element::i64, Shape{3}, {0});
        const auto end = std::make_shared<opset8::ScatterUpdate>(default_end, axes, stop, zero);

        const auto default_stride = opset8::Constant::create(element::i64, Shape{3}, {1});
        const auto stride = std::make_shared<opset8::ScatterUpdate>(default_stride, axes, step, zero);
        std::vector<int64_t> begin_end_mask = {1, 0, 0};
        auto strided_slice =
            std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_axes_const_unsorted) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{2});

        auto axes = opset8::Constant::create(element::i64, Shape{2}, {3, 1});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto start = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto stop = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{2});

        auto zero = opset8::Constant::create(element::i32, Shape{1}, {0});
        auto axes = opset8::Constant::create(element::i64, Shape{2}, {3, 1});

        const auto default_begin = opset8::Constant::create(element::i64, Shape{4}, {0});
        const auto begin = std::make_shared<opset8::ScatterUpdate>(default_begin, axes, start, zero);

        const auto default_end = opset8::Constant::create(element::i64, Shape{4}, {0});
        const auto end = std::make_shared<opset8::ScatterUpdate>(default_end, axes, stop, zero);

        const auto default_stride = opset8::Constant::create(element::i64, Shape{4}, {1});
        const auto stride = std::make_shared<opset8::ScatterUpdate>(default_stride, axes, step, zero);

        std::vector<int64_t> begin_end_mask = {1, 0, 1, 0};
        auto strided_slice =
            std::make_shared<opset8::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_axes_const_negative_sorted) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{4});

        auto axes = opset8::Constant::create(element::i64, Shape{4}, {0, -3, 2, -1});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto stride = std::make_shared<opset8::Parameter>(element::i64, Shape{4});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<opset8::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data, begin, end, stride});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_dyn_shape_axes_const_negative_unsorted) {
    {
        auto data_shape = ov::PartialShape{ov::Dimension(-1), ov::Dimension(2, 6), 4, ov::Dimension(-1)};
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto begin = std::make_shared<opset8::Parameter>(element::i64, PartialShape{ov::Dimension(-1)});
        auto end = std::make_shared<opset8::Parameter>(element::i64, PartialShape{ov::Dimension(-1)});
        auto step = std::make_shared<opset8::Parameter>(element::i64, PartialShape{ov::Dimension(-1)});

        auto axes = opset1::Constant::create(element::i64, Shape{2}, {-1, -3});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data_shape = ov::PartialShape{ov::Dimension(-1), ov::Dimension(2, 6), 4, ov::Dimension(-1)};
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto start = std::make_shared<opset8::Parameter>(element::i64, PartialShape{ov::Dimension(-1)});
        auto stop = std::make_shared<opset8::Parameter>(element::i64, PartialShape{ov::Dimension(-1)});
        auto step = std::make_shared<opset8::Parameter>(element::i64, PartialShape{ov::Dimension(-1)});

        auto zero = opset8::Constant::create(element::i32, Shape{1}, {0});
        auto axes = opset8::Constant::create(element::i64, Shape{2}, {3, 1});

        const auto default_begin = opset8::Constant::create(element::i64, Shape{4}, {0});
        const auto begin = std::make_shared<opset8::ScatterUpdate>(default_begin, axes, start, zero);

        const auto default_end = opset8::Constant::create(element::i64, Shape{4}, {0});
        const auto end = std::make_shared<opset8::ScatterUpdate>(default_end, axes, stop, zero);

        const auto default_stride = opset8::Constant::create(element::i64, Shape{4}, {1});
        const auto stride = std::make_shared<opset8::ScatterUpdate>(default_stride, axes, step, zero);

        std::vector<int64_t> begin_end_mask = {1, 0, 1, 0};
        auto strided_slice =
            std::make_shared<opset8::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_sss_params_static_shape_axes_const_negative_unsorted) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{2});

        auto axes = opset1::Constant::create(element::i64, Shape{2}, {-1, -3});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto start = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto stop = std::make_shared<opset8::Parameter>(element::i64, Shape{2});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{2});

        auto zero = opset8::Constant::create(element::i32, Shape{1}, {0});
        auto axes = opset8::Constant::create(element::i64, Shape{2}, {3, 1});

        const auto default_begin = opset8::Constant::create(element::i64, Shape{4}, {0});
        const auto begin = std::make_shared<opset8::ScatterUpdate>(default_begin, axes, start, zero);

        const auto default_end = opset8::Constant::create(element::i64, Shape{4}, {0});
        const auto end = std::make_shared<opset8::ScatterUpdate>(default_end, axes, stop, zero);

        const auto default_stride = opset8::Constant::create(element::i64, Shape{4}, {1});
        const auto stride = std::make_shared<opset8::ScatterUpdate>(default_stride, axes, step, zero);

        std::vector<int64_t> begin_end_mask = {1, 0, 1, 0};
        auto strided_slice =
            std::make_shared<opset8::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data, start, stop, step});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_dyn_rank_axes_const_positive) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, ov::PartialShape::dynamic());
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{4});

        auto axes = opset8::Constant::create(element::i64, Shape{4}, {0, 1, 2, 3});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, ov::PartialShape::dynamic());
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto stride = std::make_shared<opset8::Parameter>(element::i64, Shape{4});

        std::vector<int64_t> begin_mask = {0, 0, 0, 0};
        std::vector<int64_t> end_mask = {0, 0, 0, 0};

        auto strided_slice = std::make_shared<opset8::StridedSlice>(data, begin, end, stride, begin_mask, end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data, begin, end, stride});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_dyn_rank_axes_const_negative) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, ov::PartialShape::dynamic());
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto end = std::make_shared<opset8::Parameter>(element::i64, Shape{4});
        auto step = std::make_shared<opset8::Parameter>(element::i64, Shape{4});

        auto axes = opset8::Constant::create(element::i64, Shape{4}, {0, -3, 2, -1});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin, end, step});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    // No transformation for negative axes and dynamic data rank
}

TEST_F(TransformationTestsF, SliceToStridedSlice_axes_param) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 4, 3, 5});
        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {-1, -1, -1, -1});
        auto step = opset8::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

        auto axes = std::make_shared<opset8::Parameter>(element::i64, Shape{4});

        auto slice = std::make_shared<opset8::Slice>(data, begin, end, step, axes);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, axes});
        manager.register_pass<ov::pass::StridedSliceOptimization>();
        manager.register_pass<pass::ConstantFolding>();
    }
    // No transformation for non-const axes input
}

TEST_F(TransformationTestsF, SliceToStridedSlice_begin_param_shape_of_use_shapes_true) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<opset8::ShapeOf>(data, element::i64);
        auto data_rank = std::make_shared<opset8::ShapeOf>(shape_of_data, element::i64);

        auto zero_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});
        auto three_const = opset8::Constant::create(element::i64, Shape{}, {3});

        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{1});
        auto end = std::make_shared<opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<opset8::Range>(zero_const, one_const, one_const, element::i64);

        auto slice = std::make_shared<opset8::Slice>(shape_of_data, begin, end, step, axes);
        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin});

        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>(true);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});
        auto three_const = opset8::Constant::create(element::i64, Shape{}, {3});

        auto data = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{1});
        auto end = std::make_shared<opset8::Broadcast>(three_const, begin);
        auto stride = std::make_shared<opset8::Broadcast>(one_const, begin);

        std::vector<int64_t> begin_end_mask = {0};
        auto strided_slice =
            std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{begin});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_begin_param_shape_of_use_shapes_false) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<opset8::ShapeOf>(data, element::i64);
        auto data_rank = std::make_shared<opset8::ShapeOf>(shape_of_data, element::i64);

        auto zero_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});
        auto three_const = opset8::Constant::create(element::i64, Shape{}, {3});

        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{1});
        auto end = std::make_shared<opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<opset8::Range>(zero_const, one_const, one_const, element::i64);

        auto slice = std::make_shared<opset8::Slice>(shape_of_data, begin, end, step, axes);
        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data, begin});

        manager.register_pass<pass::ConstantFolding>();
        manager.register_pass<ov::pass::SliceToStridedSlice>(false);
        manager.register_pass<ov::pass::StridedSliceOptimization>(false);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});
        auto three_const = opset8::Constant::create(element::i64, Shape{}, {3});

        auto data = opset8::Constant::create(element::i64, Shape{4}, {2, 3, 4, 5});
        auto begin = std::make_shared<opset8::Parameter>(element::i64, Shape{1});
        auto end = std::make_shared<opset8::Broadcast>(three_const, begin);
        auto stride = std::make_shared<opset8::Broadcast>(one_const, begin);

        std::vector<int64_t> begin_end_mask = {0};
        auto strided_slice =
            std::make_shared<opset1::StridedSlice>(data, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{begin});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_const_fold_params_slice_shape_of_use_shapes_true) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<opset8::ShapeOf>(data, element::i64);
        auto data_rank = std::make_shared<opset8::ShapeOf>(shape_of_data, element::i64);

        auto zero_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});
        auto three_const = opset8::Constant::create(element::i64, Shape{}, {3});

        auto begin = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto end = std::make_shared<opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<opset8::Range>(zero_const, one_const, one_const, element::i64);

        auto slice = std::make_shared<opset8::Slice>(shape_of_data, begin, end, step, axes);
        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data});

        manager.register_pass<pass::ConstantFolding>();
        manager.register_pass<ov::pass::StridedSliceOptimization>(true);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto sliced_const = opset8::Constant::create(element::i64, Shape{2}, {3, 4});
        model_ref = std::make_shared<ov::Model>(NodeVector{sliced_const}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_const_fold_params_slice_shape_of_use_shapes_false) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto shape_of_data = std::make_shared<opset8::ShapeOf>(data, element::i64);
        auto data_rank = std::make_shared<opset8::ShapeOf>(shape_of_data, element::i64);

        auto zero_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});
        auto three_const = opset8::Constant::create(element::i64, Shape{}, {3});

        auto begin = opset8::Constant::create(element::i64, Shape{1}, {1});
        auto end = std::make_shared<opset8::Broadcast>(three_const, begin);
        auto step = std::make_shared<opset8::Broadcast>(one_const, begin);

        auto axes = std::make_shared<opset8::Range>(zero_const, one_const, one_const, element::i64);

        auto slice = std::make_shared<opset8::Slice>(shape_of_data, begin, end, step, axes);
        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data});

        manager.register_pass<pass::ConstantFolding>();
        manager.register_pass<ov::pass::StridedSliceOptimization>(false);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto sliced_const = opset8::Constant::create(element::i64, Shape{2}, {3, 4});
        model_ref = std::make_shared<ov::Model>(NodeVector{sliced_const}, ParameterVector{});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_slice_all_use_shapes_true) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto relu = std::make_shared<opset8::Relu>(data);

        auto shape_of_data = std::make_shared<opset8::ShapeOf>(relu, element::i64);
        auto data_rank = std::make_shared<opset8::ShapeOf>(shape_of_data, element::i64);

        auto zero_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});

        auto begin = std::make_shared<opset8::Broadcast>(zero_const, data_rank);
        auto end = std::make_shared<opset8::Broadcast>(data_rank, data_rank);
        auto step = std::make_shared<opset8::Broadcast>(one_const, data_rank);

        auto slice = std::make_shared<opset8::Slice>(relu, begin, end, step);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data});
        manager.register_pass<ov::pass::SliceToStridedSlice>(true);
        manager.register_pass<ov::pass::StridedSliceOptimization>(true);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto relu = std::make_shared<opset8::Relu>(data);

        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {4});
        auto stride = opset8::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_end_mask = {0, 0, 0, 0};
        auto strided_slice =
            std::make_shared<opset8::StridedSlice>(relu, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, SliceToStridedSlice_slice_all_use_shapes_false) {
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto relu = std::make_shared<opset8::Relu>(data);

        auto shape_of_data = std::make_shared<opset8::ShapeOf>(relu, element::i64);
        auto data_rank = std::make_shared<opset8::ShapeOf>(shape_of_data, element::i64);

        auto zero_const = opset8::Constant::create(element::i64, Shape{}, {0});
        auto one_const = opset8::Constant::create(element::i64, Shape{}, {1});

        auto begin = std::make_shared<opset8::Broadcast>(zero_const, data_rank);
        auto end = std::make_shared<opset8::Broadcast>(data_rank, data_rank);
        auto step = std::make_shared<opset8::Broadcast>(one_const, data_rank);

        auto slice = std::make_shared<opset8::Slice>(relu, begin, end, step);

        model = std::make_shared<ov::Model>(NodeVector{slice}, ParameterVector{data});
        manager.register_pass<ov::pass::SliceToStridedSlice>(false);
        manager.register_pass<ov::pass::StridedSliceOptimization>(false);
        manager.register_pass<pass::ConstantFolding>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 3, 4, 5});
        auto relu = std::make_shared<opset8::Relu>(data);

        auto begin = opset8::Constant::create(element::i64, Shape{4}, {0});
        auto end = opset8::Constant::create(element::i64, Shape{4}, {4});
        auto stride = opset8::Constant::create(element::i64, Shape{4}, {1});

        std::vector<int64_t> begin_end_mask = {0, 0, 0, 0};
        auto strided_slice =
            std::make_shared<opset8::StridedSlice>(relu, begin, end, stride, begin_end_mask, begin_end_mask);

        model_ref = std::make_shared<ov::Model>(NodeVector{strided_slice}, ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

ov::Output<ov::Node> make_slice(const ov::Output<ov::Node>& out,
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

ov::OutputVector make_vsplit(const ov::Output<ov::Node>& out,
                             const int64_t& axis,
                             const std::vector<int64_t>& split_length) {
    return std::make_shared<ov::op::v1::VariadicSplit>(
               out,
               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis}),
               ov::op::v0::Constant::create(ov::element::i64, ov::Shape{split_length.size()}, split_length))
        ->outputs();
}

TEST_F(TransformationTestsF, GroupedSliceToVSplit) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, -1, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        auto slice_0 = make_slice(relu, 0, 1, 1, -3);
        auto slice_1 = make_slice(relu, 1, 2, 1, 1);
        auto slice_2 = make_slice(relu, 2, 7, 1, 1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_2, slice_1}, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::GroupedSliceToVSplitOptimization>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, -1, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        auto vsplit = make_vsplit(relu, 1, {1, 1, 1});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[2], vsplit[1]}, 1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedSliceToVSplitChained) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 10, -1, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        // dimension == 10 on axis == 1 aka -3

        auto slice_0 = make_slice(relu, 0, 3, 1, -3);  // slices 0, 1, 2
        auto slice_1 = make_slice(relu, 3, 7, 1, 1);   // slices 3, 4, 5, 6
        auto slice_2 = make_slice(relu, 7, 15, 1, 1);  // slices 7, 8, 9

        auto slice_0_0 = make_slice(slice_0, 0, 1, 1, 1);    // slices 0
        auto slice_0_1 = make_slice(slice_0, 1, 100, 1, 1);  // slices 1, 2

        auto slice_1_0 = make_slice(slice_1, 0, 2, 1, 1);  // slices 3, 4
        auto slice_1_1 = make_slice(slice_1, 2, 2, 1, 1);  // slices empty tensor
        auto slice_1_2 = make_slice(slice_1, 2, 4, 1, 1);  // slices 5, 6

        auto slice_2_0 = make_slice(slice_2, 0, 2, -1, 1);  // negative case as step is negative
        auto slice_2_1 = make_slice(slice_2, 2, 10, 1, 1);

        auto slice_2_0_0 = make_slice(slice_2, 0, 3, 1, 1);  // negative case as slices overlap
        auto slice_2_1_0 = make_slice(slice_2, 2, 10, 1, 1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0,
                                                                            slice_2,
                                                                            slice_1,
                                                                            slice_0_0,
                                                                            slice_1_0,
                                                                            slice_0_1,
                                                                            slice_1_1,
                                                                            slice_1_2,
                                                                            slice_2_0,
                                                                            slice_2_1,
                                                                            slice_2_0_0,
                                                                            slice_2_1_0},
                                                           1);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::GroupedSliceToVSplitOptimization>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 10, -1, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        auto vsplit = make_vsplit(relu, 1, {3, 4, 3});

        auto slice_0 = vsplit[0];
        auto slice_1 = vsplit[1];
        auto slice_2 = vsplit[2];

        auto vsplit_0 = make_vsplit(slice_0, 1, {1, 2});

        auto slice_0_0 = vsplit_0[0];
        auto slice_0_1 = vsplit_0[1];

        auto vsplit_1 = make_vsplit(slice_1, 1, {2, 0, 2});

        auto slice_1_0 = vsplit_1[0];
        auto slice_1_1 = vsplit_1[1];
        auto slice_1_2 = vsplit_1[2];

        auto slice_2_0 = make_slice(slice_2, 0, 2, -1, 1);  // negative case as step is negative
        auto slice_2_1 = make_slice(slice_2, 2, 10, 1, 1);

        auto slice_2_0_0 = make_slice(slice_2, 0, 3, 1, 1);  // negative case as slices overlap
        auto slice_2_1_0 = make_slice(slice_2, 2, 10, 1, 1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0,
                                                                            slice_2,
                                                                            slice_1,
                                                                            slice_0_0,
                                                                            slice_1_0,
                                                                            slice_0_1,
                                                                            slice_1_1,
                                                                            slice_1_2,
                                                                            slice_2_0,
                                                                            slice_2_1,
                                                                            slice_2_0_0,
                                                                            slice_2_1_0},
                                                           1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedSliceToVSplitSameSourceDifferentAxis) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, 10, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        // axis == 1 aka -3
        auto slice_0 = make_slice(relu, 0, 1, 1, -3);
        auto slice_1 = make_slice(relu, 1, 7, 1, 1);

        // axis == 2 aka -2
        auto slice_2 = make_slice(relu, 0, 5, 1, -2);
        auto slice_3 = make_slice(relu, 5, 10, 1, 2);

        auto concat_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_1, slice_0}, 1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_2, slice_3}, 2);

        auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{concat_0, concat_1}, 0);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat_2}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::GroupedSliceToVSplitOptimization>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 3, 10, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        auto vsplit_0 = make_vsplit(relu, 1, {1, 2});
        auto slice_0 = vsplit_0[0];
        auto slice_1 = vsplit_0[1];

        auto vsplit_1 = make_vsplit(relu, 2, {5, 5});
        auto slice_2 = vsplit_1[0];
        auto slice_3 = vsplit_1[1];

        auto concat_0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_1, slice_0}, 1);
        auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_2, slice_3}, 2);

        auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{concat_0, concat_1}, 0);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat_2}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupedSliceToVSplitNegativeStartStop) {
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 5, -1, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        auto slice_0 = make_slice(relu, -50, 1, 1, -3);
        auto slice_1 = make_slice(relu, -4, -2, 1, 1);
        auto slice_2 = make_slice(relu, -2, INT32_MAX, 1, 1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{slice_0, slice_2, slice_1}, 1);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::GroupedSliceToVSplitOptimization>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 5, -1, -1});
        auto relu = std::make_shared<ov::opset8::Relu>(data);

        auto vsplit = make_vsplit(relu, 1, {1, 2, 2});

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{vsplit[0], vsplit[2], vsplit[1]}, 1);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SliceSequenceToSingleSlice) {
    auto data_pshape = ov::PartialShape{10, 5, 5, 10};
    auto data_type = ov::element::f32;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(data_type, data_pshape);

        auto slice_0 = make_slice(data, 1, 10, 1, 0);
        auto slice_1 = make_slice(slice_0, -1, 1, -1, 1);
        auto slice_2 = make_slice(slice_1, -7, INT32_MAX, 2, 3);

        model = std::make_shared<ov::Model>(ov::OutputVector{slice_2}, ov::ParameterVector{data});
        manager.register_pass<ov::pass::SliceSequenceToSingleSlice>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(data_type, data_pshape);
        auto slice = std::make_shared<ov::op::v8::Slice>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, -1, -7}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {10, 1, INT32_MAX}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, -1, 2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 3}));
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{slice}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SliceSequenceToSingleSliceStartAsParameter) {
    auto data_pshape = ov::PartialShape{10, 5, 5, 10};
    auto data_type = ov::element::f32;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(data_type, data_pshape);

        auto start_0 = std::make_shared<ov::opset8::Parameter>(element::i64, ov::PartialShape{1});
        auto start_1 = std::make_shared<ov::opset8::Parameter>(element::i64, ov::PartialShape{1});
        auto start_2 = std::make_shared<ov::opset8::Parameter>(element::i64, ov::PartialShape{1});
        auto slice_0 =
            std::make_shared<ov::op::v8::Slice>(data,
                                                start_0,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {10}),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

        auto slice_1 =
            std::make_shared<ov::op::v8::Slice>(slice_0,
                                                start_1,
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}),
                                                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));

        auto slice_2 = std::make_shared<ov::op::v8::Slice>(
            slice_1,
            start_2,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {INT32_MAX}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}));

        model = std::make_shared<ov::Model>(ov::OutputVector{slice_2},
                                            ov::ParameterVector{data, start_0, start_1, start_2});
        manager.register_pass<ov::pass::SliceSequenceToSingleSlice>();
    }
    {
        auto data = std::make_shared<ov::opset8::Parameter>(data_type, data_pshape);
        auto start_0 = std::make_shared<ov::opset8::Parameter>(element::i64, ov::PartialShape{1});
        auto start_1 = std::make_shared<ov::opset8::Parameter>(element::i64, ov::PartialShape{1});
        auto start_2 = std::make_shared<ov::opset8::Parameter>(element::i64, ov::PartialShape{1});
        auto concat_0_1 = std::make_shared<ov::opset8::Concat>(OutputVector{start_0, start_1}, 0);
        auto concat_1_2 = std::make_shared<ov::opset8::Concat>(OutputVector{concat_0_1, start_2}, 0);
        auto slice = std::make_shared<ov::op::v8::Slice>(
            data,
            concat_1_2,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {10, 1, INT32_MAX}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, -1, 2}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 3}));
        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{slice}, ov::ParameterVector{data, start_0, start_1, start_2});
    }
}
