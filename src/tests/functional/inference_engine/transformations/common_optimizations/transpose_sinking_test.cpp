// Copyright (C) 2018-2022 Intel Corporation
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
#include <openvino/core/preprocess/pre_post_process.hpp>

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
    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::TransposeFQReduction>();
    manager.register_pass<ngraph::pass::TransposeReduction>();
    manager.register_pass<ngraph::pass::CheckUniqueNames>(unh);
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto fc = FunctionsComparator::no_default()
            .enable(FunctionsComparator::NODES)
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::CONST_VALUES);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}


INSTANTIATE_TEST_SUITE_P(TransformationTest, TransposeSinkingFQ, testing::Values(
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
    auto unh = std::make_shared<ngraph::pass::UniqueNamesHolder>();
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::TransposeReduction>();
    manager.register_pass<ngraph::pass::CheckUniqueNames>(unh);
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto fc = FunctionsComparator::no_default()
            .enable(FunctionsComparator::NODES)
            .enable(FunctionsComparator::PRECISIONS)
            .enable(FunctionsComparator::CONST_VALUES);

    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}


INSTANTIATE_TEST_SUITE_P(TransposeSinkingReduces, TransposeSinking, testing::Combine(
        testing::Values(
            TransposeReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1, 2}, true, {2, 3}, {0, 2, 3, 1}},
            TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70}, {0, 6, 1, 5, 2, 4, 3}, {1, 3, 6}, true, {6, 5, 3}, {0, 6, 1, 5, 2, 4, 3}},
            TransposeReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1, 2}, false, {2, 3}, {0, 1}},
            TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70}, {0, 6, 1, 5, 2, 4, 3}, {1, 3, 6}, false, {6, 5, 3}, {0, 1, 2, 3}},
            TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70}, {0, 6, 1, 5, 2, 4, 3}, {1, -4, 6}, false, {6, 5, 3}, {0, 1, 2, 3}},
            TransposeReduceParams{{1, 3, 240, 140}, {0, 1, 2, 3}, {0, 1, 2, -1}, false, {0, 1, 2, 3}, {}}),
        testing::Values(
            ngraph::opset6::ReduceMax::get_type_info_static(),
            ngraph::opset6::ReduceMean::get_type_info_static(),
            ngraph::opset6::ReduceMin::get_type_info_static(),
            ngraph::opset6::ReduceProd::get_type_info_static(),
            ngraph::opset6::ReduceSum::get_type_info_static(),
            ngraph::opset6::ReduceL1::get_type_info_static(),
            ngraph::opset6::ReduceL2::get_type_info_static(),
            ngraph::opset6::ReduceLogicalAnd::get_type_info_static(),
            ngraph::opset6::ReduceLogicalOr::get_type_info_static())));

INSTANTIATE_TEST_SUITE_P(TransposeSinkingSqueeze, TransposeSinking, testing::Combine(
        testing::Values(
            TransposeReduceParams{{2, 3, 1, 1}, {0, 2, 3, 1}, {1, 2}, false, {2, 3}, {0, 1}},
            TransposeReduceParams{{10, 20, 30, 1, 50, 1, 1}, {0, 6, 1, 5, 2, 4, 3}, {1, 3, 6}, false, {6, 5, 3}, {0, 1, 2, 3}}),
        testing::Values(
            ngraph::opset6::Squeeze::get_type_info_static())));

TEST_F(TransformationTestsF, TransposeFuseEliminatesTranspose) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2 });
        auto tr1_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 5 }, { 0, 2, 3, 4, 1 });
        auto transpose1 = std::make_shared<ngraph::opset6::Transpose>(input, tr1_order);
        auto tr2_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 5 }, { 0, 4, 1, 2, 3 });
        auto transpose2 = std::make_shared<ngraph::opset6::Transpose>(transpose1, tr2_order);
        auto add_const = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1 });
        auto add = std::make_shared<ngraph::opset6::Add>(transpose2, add_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::TransposeFuse>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2 });
        auto add_const = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1 });
        auto add = std::make_shared<ngraph::opset6::Add>(input, add_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, TransposeFuses) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto tr1_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 1, 2, 3, 4 });
        auto transpose1 = std::make_shared<ngraph::opset6::Transpose>(input, tr1_order);
        auto tr2_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 1, 3, 4, 2, 5 });
        auto transpose2 = std::make_shared<ngraph::opset6::Transpose>(transpose1, tr2_order);
        auto result = std::make_shared<ngraph::opset6::Result>(transpose2);
        result->set_layout("NC...");

        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::TransposeFuse>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto tr_order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 2, 3, 1, 4 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, tr_order);
        auto result = std::make_shared<ngraph::opset6::Result>(transpose);
        result->set_layout("NC...");

        function_ref = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, TransposeReduceNegative) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 64});
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 0, 2, 1});
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {-1});
        auto reduce_mean = std::make_shared<ngraph::opset6::ReduceMean>(transpose, axes, true);
        auto sub = std::make_shared<opset6::Subtract>(transpose, reduce_mean);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ sub }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::TransposeReduction>();
    }
}

TEST_F(TransformationTestsF, TransposeConvert) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 1, 2, 3, 4 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto convert = std::make_shared<ngraph::opset6::Convert>(transpose, element::f16);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ convert }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::TransposeConvert>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto convert = std::make_shared<ngraph::opset6::Convert>(input, element::f16);
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 1, 2, 3, 4 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(convert, order);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ transpose }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, TransposeConvertNegativeConsumers) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 640, 20, 2, 2 });
        auto order = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 6 }, { 0, 5, 1, 2, 3, 4 });
        auto transpose = std::make_shared<ngraph::opset6::Transpose>(input, order);
        auto convert = std::make_shared<ngraph::opset6::Convert>(transpose, element::f16);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ convert, transpose }, ngraph::ParameterVector{ input });
        manager.register_pass<ngraph::pass::TransposeConvert>();
    }
}