// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/merge_similar_branches.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

using namespace std;
using namespace testing;
using namespace ov;
using namespace ov::element;
using namespace ov::pass;

class MergeSimilarBranchesTest : public TransformationTestsF {
public:
    MergeSimilarBranchesTest() {
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::CONSUMERS_COUNT);
    }
};

TEST_F(MergeSimilarBranchesTest, unary_nodes) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg = make_shared<Negative>(input);
        const auto abs1 = make_shared<Abs>(neg);
        const auto abs2 = make_shared<Abs>(neg);
        const auto abs3 = make_shared<Abs>(neg);
        const auto relu1 = make_shared<Relu>(abs1);
        const auto relu2 = make_shared<Relu>(abs2);
        const auto relu3 = make_shared<Relu>(abs3);
        const auto concat = make_shared<Concat>(OutputVector{relu1, relu2, relu3}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg = make_shared<Negative>(input);
        const auto abs = make_shared<Abs>(neg);
        const auto relu = make_shared<Relu>(abs);
        const auto concat = make_shared<Concat>(OutputVector{relu, relu, relu}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, binary_nodes_equal_constants) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto constant1 = Constant::create(f32, {1}, {1});
        const auto constant2 = Constant::create(f32, {1}, {1});
        const auto add1 = make_shared<Add>(input, constant1);
        const auto add2 = make_shared<Add>(input, constant2);
        const auto mul = make_shared<Multiply>(add1, add2);
        model = make_shared<Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto constant = Constant::create(f32, {1}, {1});
        const auto add = make_shared<Add>(input, constant);
        const auto mul = make_shared<Multiply>(add, add);
        model_ref = make_shared<Model>(NodeVector{mul}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, binary_nodes_same_op_node) {
    using namespace ov::opset11;
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{2, 3});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{3, 2});
        const auto new_shape = Constant::create(i64, {2}, {2, 3});
        const auto reshape = make_shared<Reshape>(input2, new_shape, false);
        const auto sub1 = make_shared<Subtract>(input1, reshape);
        const auto sub2 = make_shared<Subtract>(input1, reshape);
        const auto mul = make_shared<Multiply>(sub1, sub2);
        model = make_shared<Model>(NodeVector{mul}, ParameterVector{input1, input2});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{2, 3});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{3, 2});
        const auto new_shape = Constant::create(i64, {2}, {2, 3});
        const auto reshape = make_shared<Reshape>(input2, new_shape, false);
        const auto sub = make_shared<Subtract>(input1, reshape);
        const auto mul = make_shared<Multiply>(sub, sub);
        model_ref = make_shared<Model>(NodeVector{mul}, ParameterVector{input1, input2});
    }
}

TEST_F(MergeSimilarBranchesTest, keep_result_producers) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto abs1 = make_shared<Abs>(input);
        const auto abs2 = make_shared<Abs>(input);
        const auto relu1 = make_shared<Relu>(abs1);
        const auto relu2 = make_shared<Relu>(abs2);
        model = make_shared<Model>(NodeVector{relu1, relu2}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{});
        const auto abs = make_shared<Abs>(input);
        const auto relu1 = make_shared<Relu>(abs);
        const auto relu2 = make_shared<Relu>(abs);
        model_ref = make_shared<Model>(NodeVector{relu1, relu2}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, multiple_consumers) {
    using namespace ov::opset11;
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg1 = make_shared<Negative>(input);
        const auto neg2 = make_shared<Negative>(input);
        const auto relu1 = make_shared<Relu>(neg1);
        const auto relu2 = make_shared<Relu>(neg2);
        const auto relu3 = make_shared<Relu>(neg2);
        const auto relu4 = make_shared<Relu>(neg2);
        const auto concat = make_shared<Concat>(OutputVector{relu1, relu2, relu3, relu4}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        const auto input = make_shared<Parameter>(f32, PartialShape{1});
        const auto neg = make_shared<Negative>(input);
        const auto relu = make_shared<Relu>(neg);
        const auto concat = make_shared<Concat>(OutputVector{relu, relu, relu, relu}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, different_branches) {
    using namespace ov::opset11;
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{7});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{7});
        const auto constant1 = Constant::create(f32, {1}, {1});
        const auto constant2 = Constant::create(f32, {1}, {11});
        const auto constant3 = Constant::create(f32, {1}, {1});
        const auto add1 = make_shared<Add>(input1, constant1);
        const auto add2 = make_shared<Add>(input1, constant2);
        const auto add3 = make_shared<Add>(input1, input2);
        const auto add4 = make_shared<Add>(input1, constant3);
        const auto add5 = make_shared<Add>(input1, input2);
        const auto concat = make_shared<Concat>(OutputVector{add1, add2, add3, add4, add5}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        const auto input1 = make_shared<Parameter>(f32, PartialShape{7});
        const auto input2 = make_shared<Parameter>(f32, PartialShape{7});
        const auto constant1 = Constant::create(f32, {1}, {1});
        const auto constant2 = Constant::create(f32, {1}, {11});
        const auto add1 = make_shared<Add>(input1, constant1);
        const auto add2 = make_shared<Add>(input1, constant2);
        const auto add3 = make_shared<Add>(input1, input2);
        const auto concat = make_shared<Concat>(OutputVector{add1, add2, add3, add1, add3}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});
    }
}

TEST_F(MergeSimilarBranchesTest, different_op_versions) {
    using namespace ov::op;
    const auto input = make_shared<v0::Parameter>(f32, PartialShape{8, 128});
    const auto indices = v0::Constant::create(i64, {2}, {4, 4});
    const auto axis = v0::Constant::create(i64, {1}, {1});
    const auto gather_v7 = make_shared<v7::Gather>(input, indices, axis);
    const auto gather_v8 = make_shared<v8::Gather>(input, indices, axis);
    const auto relu1 = make_shared<v0::Relu>(gather_v7);
    const auto relu2 = make_shared<v0::Relu>(gather_v8);
    const auto add = make_shared<v1::Add>(relu1, relu2);
    model = make_shared<Model>(NodeVector{add}, ParameterVector{input});

    manager.register_pass<ov::pass::MergeSimilarBranches>();
}

TEST_F(MergeSimilarBranchesTest, different_input_nodes) {
    using namespace ov::opset11;
    const auto input1 = make_shared<Parameter>(f32, PartialShape{7});
    const auto input2 = make_shared<Parameter>(f32, PartialShape{7});
    const auto abs = make_shared<Abs>(input2);
    const auto neg = make_shared<Negative>(input2);
    const auto prelu1 = make_shared<PRelu>(input1, abs);
    const auto prelu2 = make_shared<PRelu>(input1, neg);
    const auto concat = make_shared<Concat>(OutputVector{prelu1, prelu2}, 0);
    model = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});

    manager.register_pass<ov::pass::MergeSimilarBranches>();
}
