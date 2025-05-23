// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/skip_gather_before_transpose_and_reshape.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeStaticShapeFpData) {
    PartialShape data_shape{1, 3, 12, 12};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto transpose_const = opset8::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset8::Transpose>(data, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeStaticShapeIntData) {
    PartialShape data_shape{1, 3, 12, 12};
    {
        auto data = std::make_shared<opset8::Parameter>(element::i64, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::i64, data_shape);

        auto transpose_const = opset8::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset8::Transpose>(data, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeDynamicShapeStaticBatch) {
    PartialShape data_shape{1, -1, -1, -1};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto transpose_const = opset8::Constant::create(element::i64, {4}, {0, 2, 3, 1});
        auto transpose = std::make_shared<opset8::Transpose>(data, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model_ref = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeIncorrectGatherAxis) {
    PartialShape data_shape{1, 3, 12, 12};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {2});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeDynamicBatch) {
    PartialShape data_shape{-1, -1, -1, -1};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeDynamicRank) {
    PartialShape data_shape = PartialShape::dynamic();
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeBatchNotEqualTo1) {
    PartialShape data_shape{2, 3, 12, 12};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {1}, {-1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}

TEST_F(TransformationTestsF, SkipGatherBeforeTransposeAndReshapeUnsuitableReshapePattern) {
    PartialShape data_shape{1, -1, -1, -1};
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);

        auto indices_node = opset8::Constant::create(element::i64, {}, {0});
        auto axis_node = opset8::Constant::create(element::i64, {}, {0});
        auto gather = std::make_shared<opset8::Gather>(data, indices_node, axis_node);

        auto transpose_const = opset8::Constant::create(element::i64, {3}, {1, 2, 0});
        auto transpose = std::make_shared<opset8::Transpose>(gather, transpose_const);

        auto reshape_const = opset8::Constant::create(element::i64, {2}, {0, -1});
        auto reshape = std::make_shared<opset8::Reshape>(transpose, reshape_const, true);

        model = std::make_shared<Model>(OutputVector{reshape}, ParameterVector{data});
        manager.register_pass<ov::pass::SkipGatherBeforeTransposeAndReshape>();
    }
}
