// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/pull_through_reduce.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset9_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;
using namespace opset9;

namespace {
template <typename ReduceType>
std::shared_ptr<Model> generate_unsqueeze_model(element::Type in_type,
                                                PartialShape in_shape,
                                                std::vector<int64_t> unsqueeze_axes,
                                                std::vector<int64_t> reduce_axes,
                                                bool keep_dims = false) {
    const auto input = std::make_shared<Parameter>(in_type, in_shape);
    const auto unsqueeze_axes_const = Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes);
    const auto unsqueeze = std::make_shared<Unsqueeze>(input, unsqueeze_axes_const);
    const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    const auto reduce_mean = std::make_shared<ReduceType>(unsqueeze, reduce_axes_const, keep_dims);

    return std::make_shared<Model>(OutputVector{reduce_mean}, ParameterVector{input});
}

template <typename ReduceType>
std::shared_ptr<Model> generate_unsqueeze_ref_model(element::Type in_type,
                                                    PartialShape in_shape,
                                                    std::vector<int64_t> unsqueeze_axes,
                                                    std::vector<int64_t> reduce_axes,
                                                    bool keep_dims = false) {
    const auto input = std::make_shared<Parameter>(in_type, in_shape);
    const auto unsqueeze_axes_const = Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes);
    const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    const auto reduce_mean = std::make_shared<ReduceType>(input, reduce_axes_const, keep_dims);
    const auto unsqueeze = std::make_shared<Unsqueeze>(reduce_mean, unsqueeze_axes_const);

    return std::make_shared<Model>(OutputVector{unsqueeze}, ParameterVector{input});
}

template <typename ReduceType>
std::shared_ptr<Model> generate_reshape_model(element::Type in_type,
                                              PartialShape in_shape,
                                              std::vector<int64_t> reshape_target_shape,
                                              std::vector<int64_t> reduce_axes,
                                              bool keep_dims = false,
                                              bool reshape_special_zero = false) {
    const auto input = std::make_shared<Parameter>(in_type, in_shape);
    const auto reshape_target_shape_const =
        Constant::create(element::i64, Shape{reshape_target_shape.size()}, reshape_target_shape);
    const auto reshape = std::make_shared<Reshape>(input, reshape_target_shape_const, reshape_special_zero);
    const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    const auto reduce_mean = std::make_shared<ReduceType>(reshape, reduce_axes_const, keep_dims);

    return std::make_shared<Model>(OutputVector{reduce_mean}, ParameterVector{input});
}

template <typename ReduceType>
std::shared_ptr<Model> generate_reshape_ref_model(element::Type in_type,
                                                  PartialShape in_shape,
                                                  std::vector<int64_t> reshape_target_shape,
                                                  std::vector<int64_t> reduce_axes,
                                                  bool keep_dims = false,
                                                  bool reshape_special_zero = false) {
    const auto input = std::make_shared<Parameter>(in_type, in_shape);
    const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    const auto reduce_mean = std::make_shared<ReduceType>(input, reduce_axes_const, keep_dims);
    const auto reshape_target_shape_const =
        Constant::create(element::i64, Shape{reshape_target_shape.size()}, reshape_target_shape);
    const auto reshape = std::make_shared<Reshape>(reduce_mean, reshape_target_shape_const, reshape_special_zero);

    return std::make_shared<Model>(OutputVector{reshape}, ParameterVector{input});
}
}  // namespace

struct PullUnsqueezeParams {
    element::Type in_type;
    PartialShape in_shape;
    std::vector<int64_t> unsqueeze_axes;
    std::vector<int64_t> ref_unsqueeze_axes;
    std::vector<int64_t> reduce_axes;
    std::vector<int64_t> ref_reduce_axes;
    bool keep_dims;
};

class PullUnsqueezeThroughReduceMean : public WithParamInterface<PullUnsqueezeParams>, public TransformationTestsF {};

TEST_P(PullUnsqueezeThroughReduceMean, PullUnsqueezeThroughReduceMeanPattern) {
    const auto& p = GetParam();
    {
        model =
            generate_unsqueeze_model<ReduceMean>(p.in_type, p.in_shape, p.unsqueeze_axes, p.reduce_axes, p.keep_dims);
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        model_ref = generate_unsqueeze_ref_model<ReduceMean>(p.in_type,
                                                             p.in_shape,
                                                             p.ref_unsqueeze_axes,
                                                             p.ref_reduce_axes,
                                                             p.keep_dims);
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<PullUnsqueezeParams> reduce_mean_params = {
    PullUnsqueezeParams{element::f32, {5, 10, 15}, {0}, {0}, {2}, {1}},
    // unsqueeze axes greater than reduce axes
    PullUnsqueezeParams{element::f32, {5, 10, 15, 20}, {0, 2, 3}, {0, 2, 3}, {5, 6}, {2, 3}},
    // unsqueeze axes lower than reduce axes
    PullUnsqueezeParams{element::f32, {5, 10, 15}, {3, 4, 5}, {1, 2, 3}, {0, 2}, {0, 2}},
    // unsqueeze axes between reduce axes
    PullUnsqueezeParams{element::f32, {5, 10, 15}, {1, 3, 5}, {0, 1, 2}, {0, 2, 4}, {0, 1, 2}},
    // unsqueeze axes between reduce axes 2
    PullUnsqueezeParams{element::f32, {1, 10, 1, 20}, {0, 1, 3, 4}, {0, 1, 2, 3}, {2, 5}, {0, 1}},
    // unsqueeze axes between reduce axes, keep_dims=true
    PullUnsqueezeParams{element::f32, {5, 10, 15}, {1, 3, 5}, {1, 3, 5}, {0, 2, 4}, {0, 1, 2}, true},
    // negative unsqueeze axes between negative reduce axes
    PullUnsqueezeParams{element::f32, {5, 10, 15}, {1, -3, -1}, {0, 1, 2}, {-2, 2, 0}, {0, 1, 2}},
    // dynamic input
    PullUnsqueezeParams{element::f32, {5, Dimension::dynamic(), 15}, {0}, {0}, {2}, {1}},
};

INSTANTIATE_TEST_SUITE_P(PullUnsqueezeThroughReduceMean, PullUnsqueezeThroughReduceMean, ValuesIn(reduce_mean_params));

class PullUnsqueezeThroughReduceLogicalOr : public WithParamInterface<PullUnsqueezeParams>,
                                            public TransformationTestsF {};

TEST_P(PullUnsqueezeThroughReduceLogicalOr, PullUnsqueezeThroughReduceLogicalOrPattern) {
    const auto& p = GetParam();
    {
        model = generate_unsqueeze_model<ReduceLogicalOr>(p.in_type,
                                                          p.in_shape,
                                                          p.unsqueeze_axes,
                                                          p.reduce_axes,
                                                          p.keep_dims);
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        model_ref = generate_unsqueeze_ref_model<ReduceLogicalOr>(p.in_type,
                                                                  p.in_shape,
                                                                  p.ref_unsqueeze_axes,
                                                                  p.ref_reduce_axes,
                                                                  p.keep_dims);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<PullUnsqueezeParams> reduce_logical_or_params = {
    PullUnsqueezeParams{element::boolean, {5, 10, 15}, {0}, {0}, {2}, {1}},
    // unsqueeze axes between reduce axes, keep_dims=true
    PullUnsqueezeParams{element::boolean, {1, 10, 1, 20}, {0, 1, 3, 4}, {0, 1, 3, 4}, {2, 5}, {0, 1}, true},
};

INSTANTIATE_TEST_SUITE_P(PullUnsqueezeThroughReduceLogicalOr,
                         PullUnsqueezeThroughReduceLogicalOr,
                         ValuesIn(reduce_logical_or_params));

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMeanInputHasMoreThanOneOutput) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{10, 10, 15});
    const auto split = std::make_shared<Split>(input, Constant::create(element::i64, Shape{}, {0}), 2);
    const auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {0});
    {
        const auto unsqueeze = std::make_shared<Unsqueeze>(split->output(0), unsqueeze_axes);
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
        const auto reduce_mean = std::make_shared<ReduceMean>(unsqueeze, reduce_axes);

        model = std::make_shared<Model>(OutputVector{reduce_mean, split->output(1)}, ParameterVector{input});
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {0});
        const auto reduce_mean = std::make_shared<ReduceMean>(split->output(0), reduce_axes);
        const auto unsqueeze = std::make_shared<Unsqueeze>(reduce_mean, unsqueeze_axes);

        model_ref = std::make_shared<Model>(OutputVector{unsqueeze, split->output(1)}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceSkipIfTheSameAxes) {
    model = generate_unsqueeze_model<ReduceMean>(element::f32, {5, 10, 15}, {0, 1}, {1, 2});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceSkipIfNotConstAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, Dimension::dynamic(), 15});
    const auto unsqueeze_axes = std::make_shared<Parameter>(element::i64, Shape{});
    const auto unsqueeze = std::make_shared<Unsqueeze>(input, unsqueeze_axes);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<ReduceMean>(unsqueeze, reduce_axes);

    model = std::make_shared<Model>(OutputVector{reduce_mean}, ParameterVector{input, unsqueeze_axes});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceMeanSkipIfMoreThanOneUnsqueezeConsumer) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto unsqueeze_axes = Constant::create(element::i64, Shape{1}, {0});
    const auto unsqueeze = std::make_shared<Unsqueeze>(input, unsqueeze_axes);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
    const auto reduce_mean = std::make_shared<ReduceMean>(unsqueeze, reduce_axes);
    const auto add = std::make_shared<Add>(unsqueeze, Constant::create(element::f32, Shape{}, {1}));

    model = std::make_shared<Model>(OutputVector{reduce_mean, add}, ParameterVector{input});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

struct PullReshapeParams {
    element::Type in_type;
    PartialShape in_shape;
    std::vector<int64_t> target_shape;
    std::vector<int64_t> ref_target_shape;
    std::vector<int64_t> reduce_axes;
    std::vector<int64_t> ref_reduce_axes;
    bool keep_dims;
    bool reshape_special_zero;
};

class PullReshapeThroughReduceMean : public WithParamInterface<PullReshapeParams>, public TransformationTestsF {};

TEST_P(PullReshapeThroughReduceMean, PullReshapeThroughReduceMeanPattern) {
    const auto& p = GetParam();
    {
        model = generate_reshape_model<ReduceMean>(p.in_type,
                                                   p.in_shape,
                                                   p.target_shape,
                                                   p.reduce_axes,
                                                   p.keep_dims,
                                                   p.reshape_special_zero);
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        model_ref = generate_reshape_ref_model<ReduceMean>(p.in_type,
                                                           p.in_shape,
                                                           p.ref_target_shape,
                                                           p.ref_reduce_axes,
                                                           p.keep_dims,
                                                           p.reshape_special_zero);
    }
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<PullReshapeParams> reduce_mean_reshape_params = {
    PullReshapeParams{element::f32, {5, 10, 15}, {1, 5, 10, 15}, {1, 5, 15}, {2}, {1}},
    // insert axes at the end
    PullReshapeParams{element::f32, {5, 10, 15}, {5, 10, 15, 1, 1}, {5, 1, 1}, {1, 2}, {1, 2}},
    // insert axes at the begin
    PullReshapeParams{element::f32, {5, 10, 15}, {1, 1, 1, 5, 10, 15}, {1, 1, 1, 10}, {3, 5}, {0, 2}},
    // insert axes on both sides
    PullReshapeParams{element::f32, {5, 10, 15}, {1, 1, 5, 10, 15, 1}, {1, 1, 15, 1}, {2, 3}, {0, 1}},
    // insert axes on both sides 2
    PullReshapeParams{element::f32, {1, 5, 10, 1}, {1, 1, 5, 10, 1, 1}, {1, 1, 1, 1}, {2, 3}, {1, 2}},
    // insert axes in the middle
    PullReshapeParams{element::f32, {4, 5, 6}, {1, 1, 4, 5, 6, 1, 1}, {1, 1, 1, 1}, {2, 3, 4}, {0, 1, 2}},
    PullReshapeParams{element::f32, {4, 5, 6}, {1, 1, 4, 1, 5, 6, 1, 1}, {1, 1, 1, 1, 1}, {2, 4, 5}, {0, 1, 2}},
    PullReshapeParams{element::f32, {4, 5, 6}, {1, 1, 4, 1, 1, 5, 6, 1, 1}, {1, 1, 1, 1, 1, 1}, {2, 5, 6}, {0, 1, 2}},
    PullReshapeParams{element::f32, {4, 1, 1}, {4, 1, 1, 1}, {1}, {0, 1, 2}, {0, 1, 2}},
    PullReshapeParams{element::f32, {1, 1}, {1, 1, 1}, {1, 1}, {1}, {1}},
    PullReshapeParams{element::f32, {2, 1, 3, 1}, {1, 2, 1, 1, 3, 1, 1}, {1, 1, 1}, {1, 2, 4, 5}, {0, 1, 2, 3}},
    PullReshapeParams{element::f32, {1, 3, 1}, {1, 1, 1, 3, 1, 1}, {1, 1, 1}, {0, 3, 4}, {0, 1, 2}},
    // insert axes on both sides, keep_dims=true
    PullReshapeParams{element::f32, {5, 10, 15}, {1, 5, 10, 15}, {1, 1, 1, 1}, {1, 2, 3}, {0, 1, 2}, true},
    // negative axes
    PullReshapeParams{element::f32, {5, 10, 15}, {1, 5, 10, 15, 1}, {1, 5, 1}, {-2, -3}, {1, 2}},
    // special zero true
    PullReshapeParams{element::f32, {5, 10, 15}, {5, 0, -1, 1}, {5, 10, 1}, {2}, {2}, false, true},
};

INSTANTIATE_TEST_SUITE_P(PullReshapeThroughReduceMean,
                         PullReshapeThroughReduceMean,
                         ValuesIn(reduce_mean_reshape_params));

class PullReshapeThroughReduceLogicalOr : public WithParamInterface<PullReshapeParams>, public TransformationTestsF {};

TEST_P(PullReshapeThroughReduceLogicalOr, PullReshapeThroughReduceLogicalOrPattern) {
    const auto& p = GetParam();
    {
        model = generate_reshape_model<ReduceLogicalOr>(p.in_type,
                                                        p.in_shape,
                                                        p.target_shape,
                                                        p.reduce_axes,
                                                        p.keep_dims,
                                                        p.reshape_special_zero);
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        model_ref = generate_reshape_ref_model<ReduceLogicalOr>(p.in_type,
                                                                p.in_shape,
                                                                p.ref_target_shape,
                                                                p.ref_reduce_axes,
                                                                p.keep_dims,
                                                                p.reshape_special_zero);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<PullReshapeParams> reduce_logical_or_reshape_params = {
    PullReshapeParams{element::boolean, {5, 10, 15}, {1, 5, 10, 15}, {1, 5, 15}, {2}, {1}},
    // keep_dims=true
    PullReshapeParams{element::boolean, {1, 10, 1, 20}, {1, 1, 10, 1, 20}, {1, 1, 1, 1, 1}, {2, 4}, {1, 3}, true},
};

INSTANTIATE_TEST_SUITE_P(PullReshapeThroughReduceLogicalOr,
                         PullReshapeThroughReduceLogicalOr,
                         ValuesIn(reduce_logical_or_reshape_params));

TEST_F(TransformationTestsF, PullReshapeThroughReduceMeanInputHasMoreThanOneOutput) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{10, 10, 15});
    const auto split = std::make_shared<Split>(input, Constant::create(element::i64, Shape{}, {0}), 2);
    {
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reshape = std::make_shared<Reshape>(split->output(0), target_shape, false);
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model = std::make_shared<Model>(OutputVector{reduce_mean, split->output(1)}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {0});
        const auto reduce_mean = std::make_shared<ReduceMean>(split->output(0), reduce_axes);
        const auto target_shape = Constant::create(element::i64, Shape{3}, {1, 10, 15});
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref = std::make_shared<Model>(OutputVector{reshape, split->output(1)}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMeanSkipIfDynamicInput) {
    model = generate_reshape_model<ReduceMean>(element::f32, {5, Dimension::dynamic(), 15}, {1, 5, 10, 15}, {2});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceSkipIfTheSameAxes) {
    model = generate_reshape_model<ReduceMean>(element::f32, {5, 10, 15}, {1, 5, 10, 15}, {0});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceSkipIfTheSameAxesScalarCase) {
    model = generate_reshape_model<ReduceMean>(element::f32, {}, {1}, {0});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceSkipIfTheSameAxesScalarCase2) {
    model = generate_reshape_model<ReduceMean>(element::f32, {}, {1, 1, 1}, {1});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceSkipIfReshapeDoesntUnsqueeze) {
    model = generate_reshape_model<ReduceMean>(element::f32, {1, 100, 1}, {1, 1, 100}, {2});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceSkipIfNonConstAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model = std::make_shared<Model>(OutputVector{reduce_mean}, ParameterVector{input, reduce_axes});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMeanSkipIfDynamicReshapeOutputShape) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = std::make_shared<Parameter>(element::i32, PartialShape{4});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model = std::make_shared<Model>(OutputVector{reduce_mean}, ParameterVector{input, target_shape});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMeanSkipIfMoreThanOneReshapeConsumer) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);
    const auto add = std::make_shared<Add>(reshape, Constant::create(element::f32, Shape{}, {1}));

    model = std::make_shared<Model>(OutputVector{reduce_mean, add}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}
