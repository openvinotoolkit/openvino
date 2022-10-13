// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/pull_through_reduce.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;
using namespace opset9;

namespace {
    template<typename ReduceType>
    std::shared_ptr<Model> gen_unsqueeze_model(element::Type in_type,
                                               PartialShape in_shape,
                                               std::vector<int64_t> unsqueeze_axes,
                                               std::vector<int64_t> reduce_axes,
                                               bool keep_dims = false) {
        const auto input = std::make_shared<Parameter>(in_type, in_shape);
        const auto unsqueeze_axes_const = Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes);
        const auto unsqueeze = std::make_shared<Unsqueeze>(input, unsqueeze_axes_const);
        const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
        const auto reduce_mean = std::make_shared<ReduceType>(unsqueeze, reduce_axes_const, keep_dims);

        return std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    }

    template<typename ReduceType>
    std::shared_ptr<Model> gen_unsqueeze_ref_model(element::Type in_type,
                                                   PartialShape in_shape,
                                                   std::vector<int64_t> unsqueeze_axes,
                                                   std::vector<int64_t> reduce_axes,
                                                   bool keep_dims = false) {
        const auto input = std::make_shared<Parameter>(in_type, in_shape);
        const auto unsqueeze_axes_const = Constant::create(element::i64, Shape{unsqueeze_axes.size()}, unsqueeze_axes);
        const auto reduce_axes_const = Constant::create(element::i64, Shape{reduce_axes.size()}, reduce_axes);
        const auto reduce_mean = std::make_shared<ReduceType>(input, reduce_axes_const, keep_dims);
        const auto unsqueeze = std::make_shared<Unsqueeze>(reduce_mean, unsqueeze_axes_const);

        return std::make_shared<Model>(NodeVector{unsqueeze}, ParameterVector{input});
    }
} // namespace

struct PullUnsqueezeParams {
    element::Type in_type;
    PartialShape in_shape;
    std::vector<int64_t> unsqueeze_axes;
    std::vector<int64_t> ref_unsqueeze_axes;
    std::vector<int64_t> reduce_axes;
    std::vector<int64_t> ref_reduce_axes;
    bool keep_dims;
};

class PullUnsqueezeThroughReduceMean
        : public WithParamInterface<PullUnsqueezeParams>,
          public TransformationTestsF {
};

TEST_P(PullUnsqueezeThroughReduceMean, PullUnsqueezeThroughReduceMeanPattern) {
    const auto& p = GetParam();
    {
        model = gen_unsqueeze_model<ReduceMean>(p.in_type, p.in_shape, p.unsqueeze_axes, p.reduce_axes, p.keep_dims);
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        model_ref = gen_unsqueeze_ref_model<ReduceMean>(p.in_type, p.in_shape, p.ref_unsqueeze_axes, p.ref_reduce_axes, p.keep_dims);
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

class PullUnsqueezeThroughReduceLogicalOr
        : public WithParamInterface<PullUnsqueezeParams>,
          public TransformationTestsF {
};

TEST_P(PullUnsqueezeThroughReduceLogicalOr, PullUnsqueezeThroughReduceLogicalOrPattern) {
    const auto& p = GetParam();
    {
        model = gen_unsqueeze_model<ReduceLogicalOr>(p.in_type, p.in_shape, p.unsqueeze_axes, p.reduce_axes, p.keep_dims);
        manager.register_pass<pass::PullUnsqueezeThroughReduce>();
    }
    {
        model_ref = gen_unsqueeze_ref_model<ReduceLogicalOr>(p.in_type, p.in_shape, p.ref_unsqueeze_axes, p.ref_reduce_axes, p.keep_dims);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

static const std::vector<PullUnsqueezeParams> reduce_logical_or_params = {
    PullUnsqueezeParams{element::boolean, {5, 10, 15}, {0}, {0}, {2}, {1}},
    // unsqueeze axes between reduce axes, keep_dims=true
    PullUnsqueezeParams{element::boolean, {1, 10, 1, 20}, {0, 1, 3, 4}, {0, 1, 3, 4}, {2, 5}, {0, 1}, true},
};

INSTANTIATE_TEST_SUITE_P(PullUnsqueezeThroughReduceLogicalOr, PullUnsqueezeThroughReduceLogicalOr, ValuesIn(reduce_logical_or_params));

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceSkipIfTheSameAxes) {
    model = gen_unsqueeze_model<ReduceMean>(element::f32, {5, 10, 15}, {0, 1}, {1, 2});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

TEST_F(TransformationTestsF, PullUnsqueezeThroughReduceSkipIfNotConstAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, Dimension::dynamic(), 15});
    const auto unsqueeze_axes = std::make_shared<Parameter>(element::i64, Shape{});
    const auto unsqueeze = std::make_shared<Unsqueeze>(input, unsqueeze_axes);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<ReduceMean>(unsqueeze, reduce_axes);

    model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input, unsqueeze_axes});
    manager.register_pass<pass::PullUnsqueezeThroughReduce>();
}

//TODO

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = Constant::create(element::i64, Shape{3}, {1, 5, 15});
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceLogicalOr) {
    const auto input = std::make_shared<Parameter>(element::boolean, PartialShape{5, 10, 15});
    {
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
        const auto reduce_or = std::make_shared<ReduceLogicalOr>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_or}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {1});
        const auto reduce_or = std::make_shared<ReduceLogicalOr>(input, reduce_axes);
        const auto target_shape = Constant::create(element::i64, Shape{3}, {1, 5, 15});
        const auto reshape = std::make_shared<Reshape>(reduce_or, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesAtTheEnd) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto reduce_axes = Constant::create(element::i64, Shape{2}, {1, 2});
    {
        const auto target_shape = Constant::create(element::i64, Shape{5}, {5, 10, 15, 1, 1});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto target_shape = Constant::create(element::i64, Shape{3}, {5, 1, 1});
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesAtTheBegin) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {3, 5});
        const auto target_shape = Constant::create(element::i64, Shape{6}, {1, 1, 1, 5, 10, 15});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {0, 2});
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 1, 1, 10});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesOnBothSides) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = Constant::create(element::i64, Shape{6}, {1, 1, 5, 10, 15, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {2, 3});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {0, 1});
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 1, 15, 1});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_InsertAxesOnBothSides_2) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{1, 5, 10, 1});
    {
        const auto target_shape = Constant::create(element::i64, Shape{6}, {1, 1, 5, 10, 1, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {2, 3});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {1, 2});
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_KeepDims) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
        const auto reduce_axes = Constant::create(element::i64, Shape{3}, {1, 2, 3});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes, true);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{3}, {0, 1, 2});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes, true);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceLogicalOr_KeepDims) {
    const auto input = std::make_shared<Parameter>(element::boolean, PartialShape{1, 10, 1, 20});
    {
        const auto target_shape = Constant::create(element::i64, Shape{5}, {1, 1, 10, 1, 20});
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {2, 4});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_or = std::make_shared<ReduceLogicalOr>(reshape, reduce_axes, true);

        model =  std::make_shared<Model>(NodeVector{reduce_or}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = Constant::create(element::i64, Shape{5}, {1, 1, 1, 1, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {1, 3});
        const auto reduce_or = std::make_shared<ReduceLogicalOr>(input, reduce_axes, true);
        const auto reshape = std::make_shared<Reshape>(reduce_or, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_NegativeAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = Constant::create(element::i64, Shape{5}, {1, 5, 10, 15, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {-2, -3});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = Constant::create(element::i64, Shape{3}, {1, 5, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{2}, {1, 2});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, false);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SpecialZeroTrue) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    {
        const auto target_shape = Constant::create(element::i64, Shape{4}, {5, 0, -1, 1});
        const auto reshape = std::make_shared<Reshape>(input, target_shape, true);
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

        model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
        manager.register_pass<pass::PullReshapeThroughReduce>();
    }
    {
        const auto target_shape = Constant::create(element::i64, Shape{3}, {5, 10, 1});
        const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
        const auto reduce_mean = std::make_shared<ReduceMean>(input, reduce_axes);
        const auto reshape = std::make_shared<Reshape>(reduce_mean, target_shape, true);

        model_ref =  std::make_shared<Model>(NodeVector{reshape}, ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_SkipIfDynamicInput) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, Dimension::dynamic(), 15});
    const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduceMean_SkipIfDynamicReshapeOutputShape) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = std::make_shared<Parameter>(element::i32, PartialShape{4});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {2});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input, target_shape});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SkipIfNotConstAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = std::make_shared<Parameter>(element::i64, PartialShape{});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input, reduce_axes});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SkipIfTheSameAxes) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = Constant::create(element::i64, Shape{4}, {1, 5, 10, 15});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {0});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}

TEST_F(TransformationTestsF, PullReshapeThroughReduce_SkipIfInsertAxesInTheMiddle) {
    const auto input = std::make_shared<Parameter>(element::f32, PartialShape{5, 10, 15});
    const auto target_shape = Constant::create(element::i64, Shape{4}, {5, 10, 1, 15});
    const auto reshape = std::make_shared<Reshape>(input, target_shape, false);
    const auto reduce_axes = Constant::create(element::i64, Shape{}, {0});
    const auto reduce_mean = std::make_shared<ReduceMean>(reshape, reduce_axes);

    model =  std::make_shared<Model>(NodeVector{reduce_mean}, ParameterVector{input});
    manager.register_pass<pass::PullReshapeThroughReduce>();
}
