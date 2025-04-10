// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/shared_ops_optimization.hpp"
#include "transformations/common_optimizations/simplify_shape_of_sub_graph.hpp"

using namespace ov;
using namespace ov::op;

namespace {

using TensorType = ov::element::Type_t;
using TensorShape = ov::Shape;

class EliminateUnsqueezeGatherTest : public TransformationTestsF,
                                     public testing::WithParamInterface<std::tuple<TensorType, TensorShape, size_t>> {
public:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& parameters = GetParam();
        const auto& inType = std::get<0>(parameters);
        const auto& inShape = std::get<1>(parameters);
        const auto& axis = std::get<2>(parameters);
        model = transform(inShape, inType, axis);
        model_ref = reference(inShape, inType);
        manager.register_pass<pass::EliminateUnsqueezeGather>();
    }

protected:
    static std::shared_ptr<Model> transform(const TensorShape& inShape, const TensorType& inType, size_t axis) {
        const auto parameter = std::make_shared<v0::Parameter>(inType, inShape);
        const auto unsqueeze =
            std::make_shared<v0::Unsqueeze>(parameter, v0::Constant::create(element::i64, Shape{1}, {axis}));
        const auto gather = std::make_shared<v1::Gather>(unsqueeze,
                                                         v0::Constant::create(element::i64, Shape{1}, {0}),
                                                         v0::Constant::create(element::i64, Shape{1}, {axis}));
        const auto relu = std::make_shared<v0::Relu>(gather);
        return std::make_shared<Model>(NodeVector{relu}, ParameterVector{parameter}, "Actual");
    }

    static std::shared_ptr<Model> reference(const TensorShape& inShape, const TensorType& inType) {
        const auto parameter = std::make_shared<v0::Parameter>(inType, inShape);
        const auto relu = std::make_shared<v0::Relu>(parameter);
        return std::make_shared<Model>(NodeVector{relu}, ParameterVector{parameter}, "Reference");
    }
};

TEST_P(EliminateUnsqueezeGatherTest, CompareFunctions) {}

INSTANTIATE_TEST_SUITE_P(
    smoke_NGraph,
    EliminateUnsqueezeGatherTest,
    testing::Combine(testing::Values(element::f16, element::f32, element::i32, element::i64, element::u8),
                     testing::Values(TensorShape{3, 128, 256}),
                     testing::Values(0, 1, 2, 3)));

}  // namespace

TEST_F(TransformationTestsF, GatherUnsqueeze) {
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<v8::Gather>(data, indices, axis);
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(gather, axis);
        const auto relu = std::make_shared<v0::Relu>(unsqueeze);
        model = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data, indices});
        manager.register_pass<pass::EliminateGatherUnsqueeze>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto updated_indices =
            std::make_shared<v1::Reshape>(indices, v0::Constant::create(element::i32, {1}, {1}), false);
        auto gather = std::make_shared<v8::Gather>(data, updated_indices, axis);
        const auto relu = std::make_shared<v0::Relu>(gather);
        model_ref = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data, indices});
    }
}

TEST_F(TransformationTestsF, GatherUnsqueezeReshape) {
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<v8::Gather>(data, indices, axis);
        auto unsqueeze =
            std::make_shared<v1::Reshape>(gather, v0::Constant::create(element::i32, Shape{1}, {1}), false);
        const auto relu = std::make_shared<v0::Relu>(unsqueeze);
        model = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data, indices});
        manager.register_pass<pass::EliminateGatherUnsqueeze>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto updated_indices =
            std::make_shared<v1::Reshape>(indices, v0::Constant::create(element::i32, {1}, {1}), false);
        auto gather = std::make_shared<v8::Gather>(data, updated_indices, axis);
        const auto relu = std::make_shared<v0::Relu>(gather);
        model_ref = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data, indices});
    }
}

TEST_F(TransformationTestsF, GatherUnsqueezeMul) {
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<v8::Gather>(data, indices, axis);

        auto scalar = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto bea = std::make_shared<v1::Multiply>(gather, scalar);

        auto unsqueeze = std::make_shared<v0::Unsqueeze>(bea, axis);
        const auto relu = std::make_shared<v0::Relu>(unsqueeze);
        model = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data, indices, scalar});
        manager.register_pass<pass::EliminateGatherUnsqueeze>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto updated_indices =
            std::make_shared<v1::Reshape>(indices, v0::Constant::create(element::i32, {1}, {1}), false);
        auto gather = std::make_shared<v8::Gather>(data, updated_indices, axis);

        auto scalar = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto bea = std::make_shared<v1::Multiply>(gather, scalar);

        const auto relu = std::make_shared<v0::Relu>(bea);
        model_ref = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data, indices, scalar});
    }
}

TEST_F(TransformationTestsF, GatherUnsqueezesMul) {
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<v8::Gather>(data, indices, axis);

        auto scalar = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto bea = std::make_shared<v1::Multiply>(gather, scalar);

        auto unsqueeze_0 = std::make_shared<v0::Unsqueeze>(bea, axis);
        auto unsqueeze_1 = std::make_shared<v0::Unsqueeze>(bea, axis);
        auto unsqueeze_2 = std::make_shared<v0::Unsqueeze>(bea, axis);

        auto concat = std::make_shared<v0::Concat>(OutputVector{unsqueeze_0, unsqueeze_1, unsqueeze_2}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data, indices, scalar});
        manager.register_pass<pass::SimplifyShapeOfSubGraph>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{}, {0});

        auto updated_indices =
            std::make_shared<v1::Reshape>(indices, v0::Constant::create(element::i32, {1}, {1}), false);
        auto gather = std::make_shared<v8::Gather>(data, updated_indices, axis);

        auto scalar = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto bea = std::make_shared<v1::Multiply>(gather, scalar);

        auto concat = std::make_shared<v0::Concat>(OutputVector{bea, bea, bea}, 0);

        model_ref = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data, indices, scalar});
    }
}
TEST_F(TransformationTestsF, GatherUnsqueezes) {
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{1}, {0});

        auto gather = std::make_shared<v8::Gather>(data, indices, axis);

        auto unsqueeze_0 = std::make_shared<v0::Unsqueeze>(gather, axis);
        auto unsqueeze_1 = std::make_shared<v0::Unsqueeze>(gather, axis);
        auto unsqueeze_2 = std::make_shared<v0::Unsqueeze>(gather, axis);

        auto concat = std::make_shared<v0::Concat>(OutputVector{unsqueeze_0, unsqueeze_1, unsqueeze_2, axis}, 0);

        model = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data, indices});
        manager.register_pass<pass::SharedOpOptimization>();
        manager.register_pass<pass::EliminateGatherUnsqueeze>();
    }
    {
        auto data = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1});
        auto indices = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{});
        auto axis = v0::Constant::create(element::i32, Shape{1}, {0});

        auto updated_indices =
            std::make_shared<v1::Reshape>(indices, v0::Constant::create(element::i32, {1}, {1}), false);
        auto gather = std::make_shared<v8::Gather>(data, updated_indices, axis);

        auto concat = std::make_shared<v0::Concat>(OutputVector{gather, gather, gather, axis}, 0);

        model_ref = std::make_shared<Model>(OutputVector{concat}, ParameterVector{data, indices});
    }
}
