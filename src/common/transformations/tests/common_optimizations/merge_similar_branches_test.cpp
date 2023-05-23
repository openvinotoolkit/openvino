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
        auto input = make_shared<Parameter>(f32, PartialShape{1});
        auto neg = make_shared<Negative>(input);
        auto abs1 = make_shared<Abs>(neg);
        auto abs2 = make_shared<Abs>(neg);
        auto abs3 = make_shared<Abs>(neg);
        auto relu1 = make_shared<Relu>(abs1);
        auto relu2 = make_shared<Relu>(abs2);
        auto relu3 = make_shared<Relu>(abs3);
        auto concat = make_shared<Concat>(OutputVector{relu1, relu2, relu3}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        auto input = make_shared<Parameter>(f32, PartialShape{1});
        auto neg = make_shared<Negative>(input);
        auto abs = make_shared<Abs>(neg);
        auto relu = make_shared<Relu>(abs);
        auto concat = make_shared<Concat>(OutputVector{relu, relu, relu}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, constants) {
    using namespace ov::opset11;
    {
        auto input = make_shared<Parameter>(f32, PartialShape{});
        auto constant1 = Constant::create(f32, {1}, {1});
        auto constant2 = Constant::create(f32, {1}, {1});
        auto add1 = make_shared<Add>(input, constant1);
        auto add2 = make_shared<Add>(input, constant2);
        auto mul = make_shared<Multiply>(add1, add2);
        model = make_shared<Model>(NodeVector{mul}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        auto input = make_shared<Parameter>(f32, PartialShape{});
        auto constant = Constant::create(f32, {1}, {1});
        auto add = make_shared<Add>(input, constant);
        auto mul = make_shared<Multiply>(add, add);
        model_ref = make_shared<Model>(NodeVector{mul}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, keep_result_producers) {
    using namespace ov::opset11;
    {
        auto input = make_shared<Parameter>(f32, PartialShape{});
        auto abs1 = make_shared<Abs>(input);
        auto abs2 = make_shared<Abs>(input);
        auto relu1 = make_shared<Relu>(abs1);
        auto relu2 = make_shared<Relu>(abs2);
        model = make_shared<Model>(NodeVector{relu1, relu2}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        auto input = make_shared<Parameter>(f32, PartialShape{});
        auto abs = make_shared<Abs>(input);
        auto relu1 = make_shared<Relu>(abs);
        auto relu2 = make_shared<Relu>(abs);
        model_ref = make_shared<Model>(NodeVector{relu1, relu2}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, multiple_consumers) {
    using namespace ov::opset11;
    {
        auto input = make_shared<Parameter>(f32, PartialShape{1});
        auto neg1 = make_shared<Negative>(input);
        auto neg2 = make_shared<Negative>(input);
        auto relu1 = make_shared<Relu>(neg1);
        auto relu2 = make_shared<Relu>(neg2);
        auto relu3 = make_shared<Relu>(neg2);
        auto relu4 = make_shared<Relu>(neg2);
        auto concat = make_shared<Concat>(OutputVector{relu1, relu2, relu3, relu4}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        auto input = make_shared<Parameter>(f32, PartialShape{1});
        auto neg = make_shared<Negative>(input);
        auto relu = make_shared<Relu>(neg);
        auto concat = make_shared<Concat>(OutputVector{relu, relu, relu, relu}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input});
    }
}

TEST_F(MergeSimilarBranchesTest, different_branches) {
    using namespace ov::opset11;
    {
        auto input1 = make_shared<Parameter>(f32, PartialShape{7});
        auto input2 = make_shared<Parameter>(f32, PartialShape{7});
        auto constant1 = Constant::create(f32, {1}, {1});
        auto constant2 = Constant::create(f32, {1}, {11});
        auto constant3 = Constant::create(f32, {1}, {1});
        auto add1 = make_shared<Add>(input1, constant1);
        auto add2 = make_shared<Add>(input1, constant2);
        auto add3 = make_shared<Add>(input1, input2);
        auto add4 = make_shared<Add>(input1, constant3);
        auto add5 = make_shared<Add>(input1, input2);
        auto concat = make_shared<Concat>(OutputVector{add1, add2, add3, add4, add5}, 0);
        model = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});

        manager.register_pass<ov::pass::MergeSimilarBranches>();
    }
    {
        auto input1 = make_shared<Parameter>(f32, PartialShape{7});
        auto input2 = make_shared<Parameter>(f32, PartialShape{7});
        auto constant1 = Constant::create(f32, {1}, {1});
        auto constant2 = Constant::create(f32, {1}, {11});
        auto add1 = make_shared<Add>(input1, constant1);
        auto add2 = make_shared<Add>(input1, constant2);
        auto add3 = make_shared<Add>(input1, input2);
        auto concat = make_shared<Concat>(OutputVector{add1, add2, add3, add1, add3}, 0);
        model_ref = make_shared<Model>(NodeVector{concat}, ParameterVector{input1, input2});
    }
}
