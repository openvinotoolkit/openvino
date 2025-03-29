// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/split_squeeze_concat_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, SplitSqueezeConcatFusion) {
    size_t num_splits = 4;

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {2});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(false);
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto transpose_order = opset7::Constant::create(element::i64, Shape{6}, {0, 1, 3, 4, 2, 5});
        auto transpose = std::make_shared<opset7::Transpose>(input, transpose_order);
        auto reshape_shape =
            opset7::Constant::create<int64_t>(element::i64, Shape{5}, {1, 2, 640, 20, 2 * (int64_t)num_splits});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_shape, false);

        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SplitSqueezeConcatFusionSqueezeWithoutAxesInput) {
    size_t num_splits = 4;

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{3, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i));
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(false);
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{3, 2, num_splits, 640, 20, 2});
        auto transpose_order = opset7::Constant::create(element::i64, Shape{6}, {0, 1, 3, 4, 2, 5});
        auto transpose = std::make_shared<opset7::Transpose>(input, transpose_order);
        auto reshape_shape =
            opset7::Constant::create<int64_t>(element::i64, Shape{5}, {3, 2, 640, 20, 2 * (int64_t)num_splits});
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_shape, false);

        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SplitSqueezeConcatFusionNegativeCaseNotAllSplitOutputsGoToSqueeze) {
    size_t num_splits = 4;

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits - 1);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {2});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(false);
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits - 1);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {2});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SplitSqueezeConcatFusionNegativeCaseSplitOutputsGoInDifferentOrder) {
    size_t num_splits = 4;

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {2});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        std::swap(squeeze_vec[1], squeeze_vec[2]);

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(false);
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {2});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        std::swap(squeeze_vec[1], squeeze_vec[2]);

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SplitSqueezeConcatFusionNegativeCaseSplitAxisDifferentFromSqueezeAxis) {
    size_t num_splits = 4;

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {0});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(false);
    }

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            auto squeeze_axis = opset7::Constant::create(element::i64, Shape{1}, {0});
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i), squeeze_axis)->output(0);
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 4);

        model_ref = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SplitSqueezeConcatFusionNegativeSqueezeWithoutAxesInputMultipleUnitDimensions) {
    size_t num_splits = 4;

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 2, num_splits, 640, 20, 2});
        auto split_axis = opset7::Constant::create(element::i64, Shape{}, {2});
        auto split = std::make_shared<opset7::Split>(input, split_axis, num_splits);
        OutputVector squeeze_vec(num_splits);

        for (size_t i = 0; i < squeeze_vec.size(); i++) {
            squeeze_vec[i] = std::make_shared<opset7::Squeeze>(split->output(i));
        }

        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, 3);

        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(false);
    }
}

struct SplitReshapeConcatFusionParam {
    int num_splits;
    int split_axis;
    Shape input_shape;
    std::vector<int> reshaped_shape;
    int concat_axis;
    std::vector<int> transpose_order;
    bool can_fuse;
};

class SplitReshapeConcatFusion : public TransformationTestsF,
                                 public testing::WithParamInterface<SplitReshapeConcatFusionParam> {};

TEST_P(SplitReshapeConcatFusion, SplitSqueezeConcatFusion) {
    auto params = GetParam();
    ASSERT_EQ(0, params.input_shape[params.split_axis] % params.num_splits);

    {
        auto input = std::make_shared<opset7::Parameter>(element::f32, params.input_shape);
        auto split_axis_node = opset7::Constant::create(element::i64, Shape{}, {params.split_axis});
        auto split = std::make_shared<opset7::Split>(input, split_axis_node, params.num_splits);
        OutputVector squeeze_vec;
        squeeze_vec.reserve(params.num_splits);
        auto reshaped_shape_node =
            opset7::Constant::create(element::i32, Shape{params.reshaped_shape.size()}, params.reshaped_shape);
        for (int i = 0; i < params.num_splits; i++) {
            squeeze_vec.push_back(std::make_shared<opset7::Reshape>(split->output(i), reshaped_shape_node, true));
        }
        auto concat = std::make_shared<opset7::Concat>(squeeze_vec, params.concat_axis);
        model = std::make_shared<ov::Model>(NodeVector{concat}, ParameterVector{input});
        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>(true);
    }

    if (!params.can_fuse) {
        model_ref = model->clone();
    } else {
        auto input = std::make_shared<opset7::Parameter>(element::f32, params.input_shape);
        auto transpose_order_node =
            opset7::Constant::create(element::i64, Shape{params.transpose_order.size()}, params.transpose_order);
        auto transpose = std::make_shared<opset7::Transpose>(input, transpose_order_node);
        auto reshape_shape = params.input_shape;
        reshape_shape.erase(reshape_shape.begin() + params.split_axis);
        reshape_shape[params.concat_axis] *= params.num_splits;
        auto reshape_shape_node = opset7::Constant::create(element::i64, Shape{reshape_shape.size()}, reshape_shape);
        auto reshape = std::make_shared<opset7::Reshape>(transpose, reshape_shape_node, false);

        model_ref = std::make_shared<ov::Model>(NodeVector{reshape}, ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static std::vector<SplitReshapeConcatFusionParam> split_reshape_concat_fusion_params{
    {4, 2, Shape{3, 1, 4, 1, 5}, {3, 1, 1, 5}, 1, {0, 2, 1, 3, 4}, true},
    {4, 0, Shape{4, 6, 5}, {6, 5}, 1, {1, 0, 2}, true},
    {5, 2, Shape{4, 6, 5}, {4, 6}, 0, {2, 0, 1}, true},
    {2, 2, Shape{3, 1, 4, 5}, {3, 2, 5}, 1, {0, 2, 1, 3}, false},
    {2, 1, Shape{3, 2, 3, 4, 5}, {3, 3, 5, 4}, 1, {0, 2, 1, 3, 4}, false},
    {4, 2, Shape{3, 1, 4, 1, 5}, {3, 1, 5}, 1, {0, 2, 1, 3, 4}, false},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         SplitReshapeConcatFusion,
                         testing::ValuesIn(split_reshape_concat_fusion_params));
