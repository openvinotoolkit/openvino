// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

struct TransposeFQReduceParams {
    // given params
    PartialShape transpose_input_shape;
    std::vector<int32_t> transpose_order;
    Shape il, ih, ol, oh;
    std::vector<int32_t> reduce_axes;
    bool reduce_keep_dims;

    // expected params
    Shape ex_il, ex_ih, ex_ol, ex_oh;
    std::vector<int32_t>  ex_reduce_axes;
    std::vector<int32_t> ex_transpose_order;
};

class TransposeSinkingFQ : public CommonTestUtils::TestsCommon,
                         public testing::WithParamInterface<std::tuple<TransposeFQReduceParams>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());

        {
            auto input = std::make_shared<opset6::Parameter>(element::f32, test_case.transpose_input_shape);

            auto order = std::make_shared<opset6::Constant>(element::i64, Shape{test_case.transpose_order.size()}, test_case.transpose_order);
            auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);

            auto i_low = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.il, std::vector<int32_t>{0});
            auto i_high = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.ih, std::vector<int32_t>{0});
            auto o_low = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.ol, std::vector<int32_t>{0});
            auto o_high = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.oh, std::vector<int32_t>{0});
            auto fq = std::make_shared<ngraph::opset6::FakeQuantize>(transpose, i_low, i_high, o_low, o_high, 256);

            auto axes = std::make_shared<ngraph::opset6::Constant>(
                    element::i64, Shape{test_case.reduce_axes.size()}, test_case.reduce_axes);
            auto reduce = std::make_shared<ngraph::opset6::ReduceMean>(fq, axes, test_case.reduce_keep_dims);

            f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reduce}, ngraph::ParameterVector{input});
        }

        {
            auto input = std::make_shared<opset6::Parameter>(element::f32, test_case.transpose_input_shape);

            auto i_low = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.ex_il, std::vector<int32_t>{0});
            auto i_high = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.ex_ih, std::vector<int32_t>{0});
            auto o_low = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.ex_ol, std::vector<int32_t>{0});
            auto o_high = std::make_shared<ngraph::opset6::Constant>(element::i64, test_case.ex_oh, std::vector<int32_t>{0});
            auto fq = std::make_shared<ngraph::opset6::FakeQuantize>(input, i_low, i_high, o_low, o_high, 256);

            auto axes = std::make_shared<ngraph::opset6::Constant>(
                    element::i64, Shape{test_case.ex_reduce_axes.size()}, test_case.ex_reduce_axes);
            auto reduce = std::make_shared<ngraph::opset6::ReduceMean>(fq, axes, test_case.reduce_keep_dims);

            auto order = std::make_shared<opset6::Constant>(element::i64, Shape{test_case.ex_transpose_order.size()}, test_case.ex_transpose_order);
            auto transpose = std::make_shared<ngraph::opset6::Transpose>(reduce, order);

            f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{input});
        }
    }
};

TEST_P(TransposeSinkingFQ, TransposeFQReduce) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::TransposeFQReduction>();
    manager.register_pass<ngraph::pass::TransposeReduction>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}


INSTANTIATE_TEST_CASE_P(TransformationTest, TransposeSinkingFQ, testing::Values(
        TransposeFQReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1}, {3}, {1, 1, 1, 1}, {1, 1, 1, 3}, {1, 2}, true,
                                {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {2, 3}, {0, 2, 3, 1}},
        TransposeFQReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1}, {3}, {1, 1, 1, 1}, {1, 1, 1, 3}, {1, 2}, false,
                                {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {2, 3}, {0, 1}}));



struct TransposeReduceParams {
    // given params
    PartialShape transpose_input_shape;
    std::vector<int32_t> transpose_order;
    std::vector<int32_t> reduce_axes;
    bool reduction_keep_dims;

    // expected params
    std::vector<int32_t>  ex_reduce_axes;
    std::vector<int32_t> ex_transpose_order;
};

class TransposeSinking : public CommonTestUtils::TestsCommon,
                         public testing::WithParamInterface<std::tuple<TransposeReduceParams, ngraph::NodeTypeInfo>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());
        const auto& reduction_type_info = std::get<1>(GetParam());

        {
            auto input = std::make_shared<opset6::Parameter>(element::dynamic, test_case.transpose_input_shape);

            auto order = std::make_shared<opset6::Constant>(element::i64, Shape{test_case.transpose_order.size()}, test_case.transpose_order);
            auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);

            auto axes = std::make_shared<ngraph::opset6::Constant>(
                    element::i64, Shape{test_case.reduce_axes.size()}, test_case.reduce_axes);

            auto reduction = get_reduction(reduction_type_info, {transpose, axes}, test_case.reduction_keep_dims);

            f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reduction}, ngraph::ParameterVector{input});
        }

        {
            auto input = std::make_shared<opset6::Parameter>(element::dynamic, test_case.transpose_input_shape);

            auto axes = std::make_shared<ngraph::opset6::Constant>(
                    element::i64, Shape{test_case.ex_reduce_axes.size()}, test_case.ex_reduce_axes);
            auto reduction = get_reduction(reduction_type_info, {input, axes}, test_case.reduction_keep_dims);

            auto order = std::make_shared<opset6::Constant>(element::i64, Shape{test_case.ex_transpose_order.size()}, test_case.ex_transpose_order);
            auto transpose = std::make_shared<ngraph::opset6::Transpose>(reduction, order);

            f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{input});
        }
    }
private:
    std::shared_ptr<Node> get_reduction(ngraph::NodeTypeInfo reduction_type_info, const OutputVector& inputs, bool keep_dims) {
        auto reduction = ngraph::helpers::getNodeSharedPtr(reduction_type_info, inputs);
        if (auto arithmetic_reduce = std::dynamic_pointer_cast<op::util::ArithmeticReductionKeepDims>(reduction))
            arithmetic_reduce->set_keep_dims(keep_dims);
        else if (auto logical_reduce = std::dynamic_pointer_cast<op::util::LogicalReductionKeepDims>(reduction))
            logical_reduce->set_keep_dims(keep_dims);
        reduction->validate_and_infer_types();
        return reduction;
    }
};

TEST_P(TransposeSinking, TransposeReduction) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::TransposeReduction>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref, true);

ASSERT_TRUE(res.first) << res.second;
}


INSTANTIATE_TEST_CASE_P(TransposeSinkingReduces, TransposeSinking, testing::Combine(
        testing::Values(
            TransposeReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1, 2}, true, {2, 3}, {0, 2, 3, 1}},
            TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70}, {0, 6, 1, 5, 2, 4, 3}, {1, 3, 6}, true, {6, 5, 3}, {0, 6, 1, 5, 2, 4, 3}},
            TransposeReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1, 2}, false, {2, 3}, {0, 1}},
            TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70}, {0, 6, 1, 5, 2, 4, 3}, {1, 3, 6}, false, {6, 5, 3}, {0, 1, 2, 3}},
            TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70}, {0, 6, 1, 5, 2, 4, 3}, {1, -4, 6}, false, {6, 5, 3}, {0, 1, 2, 3}},
            TransposeReduceParams{{1, 3, 240, 140}, {0, 1, 2, 3}, {0, 1, 2, -1}, false, {0, 1, 2, 3}, {}}),
        testing::Values(
            ngraph::opset6::ReduceMax::type_info,
            ngraph::opset6::ReduceMean::type_info,
            ngraph::opset6::ReduceMin::type_info,
            ngraph::opset6::ReduceProd::type_info,
            ngraph::opset6::ReduceSum::type_info,
            ngraph::opset6::ReduceL1::type_info,
            ngraph::opset6::ReduceL2::type_info,
            ngraph::opset6::ReduceLogicalAnd::type_info,
            ngraph::opset6::ReduceLogicalOr::type_info)));

INSTANTIATE_TEST_CASE_P(TransposeSinkingSqueeze, TransposeSinking, testing::Combine(
        testing::Values(
            TransposeReduceParams{{2, 3, 1, 1}, {0, 2, 3, 1}, {1, 2}, false, {2, 3}, {0, 1}},
            TransposeReduceParams{{10, 20, 30, 1, 50, 1, 1}, {0, 6, 1, 5, 2, 4, 3}, {1, 3, 6}, false, {6, 5, 3}, {0, 1, 2, 3}}),
        testing::Values(
            ngraph::opset6::Squeeze::type_info)));

TEST(TransformationTests, TransposeFuseEliminatesTranspose) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2 });
        auto tr1_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 5 }, { 0, 2, 3, 4, 1 });
        auto transpose1 = std::make_shared<ngraph::opset6::Transpose>(input, tr1_order);
        auto tr2_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 5 }, { 0, 4, 1, 2, 3 });
        auto transpose2 = std::make_shared<ngraph::opset6::Transpose>(transpose1, tr2_order);
        auto add_const = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1 });
        auto add = std::make_shared<ngraph::opset6::Add>(transpose2, add_const);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::TransposeFuse>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2 });
        auto add_const = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1 });
        auto add = std::make_shared<ngraph::opset6::Add>(input, add_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, TransposeFuses) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto tr1_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 1, 2, 3, 4 });
        auto transpose1 = std::make_shared<ngraph::opset6::Transpose>(input, tr1_order);
        auto tr2_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 1, 3, 4, 2, 5 });
        auto transpose2 = std::make_shared<ngraph::opset6::Transpose>(transpose1, tr2_order);
        auto add_const = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1 });
        auto add = std::make_shared<ngraph::opset6::Add>(transpose2, add_const);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::TransposeFuse>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto tr_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 2, 3, 1, 4 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, tr_order);
        auto add_const = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1 });
        auto add = std::make_shared<ngraph::opset6::Add>(transpose, add_const);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
