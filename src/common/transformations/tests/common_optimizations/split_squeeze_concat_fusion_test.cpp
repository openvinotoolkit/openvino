// Copyright (C) 2018-2023 Intel Corporation
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

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>();
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

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>();
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

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>();
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

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>();
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

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>();
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

        manager.register_pass<ov::pass::SplitSqueezeConcatFusion>();
    }
}
