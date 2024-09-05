// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/reduce_reshape_fusion.hpp"
#include "transformations/common_optimizations/transpose_to_reshape.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace testing;
using namespace ov;
using namespace opset9;

namespace {
template <typename ReduceType>
std::shared_ptr<Model> generate_model(element::Type in_type,
                                      PartialShape in_shape,
                                      std::vector<int64_t> reshape_target_shape,
                                      std::vector<int64_t> reduce_axes,
                                      bool reduce_keep_dims,
                                      bool reshape_special_zero) {
    const auto input = std::make_shared<Parameter>(in_type, in_shape);
    const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    const auto reduce_mean = std::make_shared<ReduceType>(input, reduce_axes_const, reduce_keep_dims);
    const auto target_shape = Constant::create(element::i64, Shape{reshape_target_shape.size()}, reshape_target_shape);
    const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, reshape_special_zero);

    return std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
}

template <typename ReduceType>
std::shared_ptr<Model> generate_ref_model(element::Type in_type,
                                          PartialShape in_shape,
                                          std::vector<int64_t> reduce_axes) {
    const auto input = std::make_shared<Parameter>(in_type, in_shape);
    const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    const auto reduce_mean = std::make_shared<ReduceType>(input, reduce_axes_const, true);

    return std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
}
}  // namespace

struct ReduceReshapeFusionParams {
    element::Type in_type;
    PartialShape in_shape;
    std::vector<int64_t> reshape_target_shape;
    std::vector<int64_t> reduce_axes;
    bool keep_dims;
    bool reshape_special_zero;
};

class ReduceReshapeFusion : public WithParamInterface<ReduceReshapeFusionParams>, public TransformationTestsF {};

TEST_P(ReduceReshapeFusion, ReduceReshapeFusionPattern) {
    const auto& p = GetParam();
    {
        model = generate_model<ReduceMean>(p.in_type,
                                           p.in_shape,
                                           p.reshape_target_shape,
                                           p.reduce_axes,
                                           p.keep_dims,
                                           p.reshape_special_zero);
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    { model_ref = generate_ref_model<ReduceMean>(p.in_type, p.in_shape, p.reduce_axes); }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<ReduceReshapeFusionParams> params = {
    ReduceReshapeFusionParams{element::f32, {5, 10, 15}, {5, 1, 15}, {1}, false, false},
    // many axes
    ReduceReshapeFusionParams{element::f32, {5, 10, 15, 20}, {5, 1, 1, 20}, {1, 2}, false, false},
    // many axes 2
    ReduceReshapeFusionParams{element::f32, {5, 10, 15, 1}, {5, 1, 1, 1}, {1, 2}, false, false},
    // special zero
    ReduceReshapeFusionParams{element::f32, {5, 10, 15, 20}, {-1, 0, 1, 1}, {2, 3}, false, true},
    // negative axes
    ReduceReshapeFusionParams{element::f32, {5, 10, 15, 20}, {5, 1, 15, 1}, {-1, -3}, false, false},
    // negative axes 2
    ReduceReshapeFusionParams{element::f32, {5, 10, 15, 20}, {5, 1, 1, 20}, {-2, -3}, false, false},
};

INSTANTIATE_TEST_SUITE_P(ReduceReshapeFusion, ReduceReshapeFusion, ValuesIn(params));

TEST_F(TransformationTestsF, ReduceOrReshapeFusion) {
    {
        model = generate_model<ReduceLogicalOr>(element::boolean, {5, 10, 15, 20}, {5, 1, 1, 20}, {1, 2}, false, false);
        manager.register_pass<pass::ReduceReshapeFusion>();
    }
    { model_ref = generate_ref_model<ReduceLogicalOr>(element::boolean, {5, 10, 15, 20}, {1, 2}); }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusionSkipIfOneInNotAxisPosition) {
    model = generate_model<ReduceMean>(element::f32, {5, 10, 15, 1}, {5, 1, 1, 1, 1}, {1, 2}, false, false);
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusionSkipIfReshapeNotCompatible) {
    model = generate_model<ReduceMean>(element::f32, {5, 10, 15, 20}, {20, 1, 1, 5}, {1, 2}, false, false);
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfReshapeRankLessThanReduceRank) {
    model = generate_model<ReduceMean>(element::f32, {5, 10, 15}, {50}, {2}, false, false);
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusion_SkipIfKeepDims) {
    model = generate_model<ReduceMean>(element::f32, {5, 10, 15}, {5, 1, 15}, {1}, true, false);
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusionSkipIfNonConstReduceAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = std::make_shared<Parameter>(element::i64, PartialShape{1});
    const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
    const auto target_shape = Constant::create(element::i64, Shape{3}, {5, 1, 15});
    const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

    model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input, reduce_axes});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusionSkipIfNonConstReshapeTargetShape) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
    const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
    const auto target_shape = std::make_shared<Parameter>(element::i64, PartialShape{3});
    const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

    model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input, target_shape});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST_F(TransformationTestsF, ReduceMeanReshapeFusionSkipIfMoreThanOneReduceConsumer) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{1, 1});
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
    const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
    const auto add = std::make_shared<Add>(reduce_mean, Constant::create(element::f32, Shape{1}, {1}));
    const auto target_shape = Constant::create(element::i64, Shape{2}, {1, 1});
    const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

    model = std::make_shared<Model>(NodeVector{reshape, add}, ParameterVector{input});
    manager.register_pass<pass::ReduceReshapeFusion>();
}

TEST(TransformationTests, ReduceMeanReshapeFusionAssertValidOutputShape) {
    const auto input = make_shared<Parameter>(element::f32, PartialShape{1, 16, 16, 24});
    const auto reduce_axes = Constant::create(element::i64, Shape{2}, {1, 2});
    const auto reduce_mean = make_shared<ReduceMean>(input, reduce_axes, false);
    const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 1, 1, 24});
    const auto reshape = make_shared<Reshape>(reduce_mean, target_shape, false);
    const auto order = Constant::create(element::i64, Shape{4}, {0, 3, 1, 2});
    const auto transpose = make_shared<Transpose>(reshape, order);

    auto model = make_shared<Model>(NodeVector{transpose}, ParameterVector{input});

    pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<pass::ReduceReshapeFusion>();
    manager.register_pass<pass::TransposeToReshape>();
    OV_ASSERT_NO_THROW(manager.run_passes(model));
}
