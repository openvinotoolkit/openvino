// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

struct TransposeFQReduceParams {
    // given params
    PartialShape transpose_input_shape;
    std::vector<int32_t> transpose_order;
    Shape il, ih, ol, oh;
    std::vector<int32_t> reduce_axes;
    bool reduce_keep_dims;

    // expected params
    Shape ex_il, ex_ih, ex_ol, ex_oh;
    std::vector<int32_t> ex_reduce_axes;
    std::vector<int32_t> ex_transpose_order;
};

class TransposeSinkingFQ : public ov::test::TestsCommon,
                           public testing::WithParamInterface<std::tuple<TransposeFQReduceParams>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());

        {
            auto input = std::make_shared<opset6::Parameter>(element::f32, test_case.transpose_input_shape);

            auto order = std::make_shared<opset6::Constant>(element::i64,
                                                            Shape{test_case.transpose_order.size()},
                                                            test_case.transpose_order);
            auto transpose = std::make_shared<opset6::Transpose>(input, order);

            auto i_low = std::make_shared<opset6::Constant>(element::i64, test_case.il, std::vector<int32_t>{0});
            auto i_high = std::make_shared<opset6::Constant>(element::i64, test_case.ih, std::vector<int32_t>{0});
            auto o_low = std::make_shared<opset6::Constant>(element::i64, test_case.ol, std::vector<int32_t>{0});
            auto o_high = std::make_shared<opset6::Constant>(element::i64, test_case.oh, std::vector<int32_t>{0});
            auto fq = std::make_shared<opset6::FakeQuantize>(transpose, i_low, i_high, o_low, o_high, 256);

            auto axes = std::make_shared<opset6::Constant>(element::i64,
                                                           Shape{test_case.reduce_axes.size()},
                                                           test_case.reduce_axes);
            auto reduce = std::make_shared<opset6::ReduceMean>(fq, axes, test_case.reduce_keep_dims);

            f = std::make_shared<ov::Model>(NodeVector{reduce}, ParameterVector{input});
        }

        {
            auto input = std::make_shared<opset6::Parameter>(element::f32, test_case.transpose_input_shape);

            auto i_low = std::make_shared<opset6::Constant>(element::i64, test_case.ex_il, std::vector<int32_t>{0});
            auto i_high = std::make_shared<opset6::Constant>(element::i64, test_case.ex_ih, std::vector<int32_t>{0});
            auto o_low = std::make_shared<opset6::Constant>(element::i64, test_case.ex_ol, std::vector<int32_t>{0});
            auto o_high = std::make_shared<opset6::Constant>(element::i64, test_case.ex_oh, std::vector<int32_t>{0});
            auto fq = std::make_shared<opset6::FakeQuantize>(input, i_low, i_high, o_low, o_high, 256);

            auto axes = std::make_shared<opset6::Constant>(element::i64,
                                                           Shape{test_case.ex_reduce_axes.size()},
                                                           test_case.ex_reduce_axes);
            auto reduce = std::make_shared<opset6::ReduceMean>(fq, axes, test_case.reduce_keep_dims);

            auto order = std::make_shared<opset6::Constant>(element::i64,
                                                            Shape{test_case.ex_transpose_order.size()},
                                                            test_case.ex_transpose_order);
            auto transpose = std::make_shared<opset6::Transpose>(reduce, order);

            f_ref = std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{input});
        }
    }
};

TEST_P(TransposeSinkingFQ, TransposeFQReduce) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::TransposeFQReduction>();
    manager.register_pass<ov::pass::TransposeReduction>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    auto fc = FunctionsComparator::no_default()
                  .enable(FunctionsComparator::NODES)
                  .enable(FunctionsComparator::PRECISIONS)
                  .enable(FunctionsComparator::CONST_VALUES);
    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

INSTANTIATE_TEST_SUITE_P(TransformationTest,
                         TransposeSinkingFQ,
                         testing::Values(TransposeFQReduceParams{{1, 3, 240, 140},
                                                                 {0, 2, 3, 1},
                                                                 {1},
                                                                 {3},
                                                                 {1, 1, 1, 1},
                                                                 {1, 1, 1, 3},
                                                                 {1, 2},
                                                                 true,
                                                                 {1, 1, 1, 1},
                                                                 {1, 3, 1, 1},
                                                                 {1, 1, 1, 1},
                                                                 {1, 3, 1, 1},
                                                                 {2, 3},
                                                                 {0, 2, 3, 1}},
                                         TransposeFQReduceParams{{1, 3, 240, 140},
                                                                 {0, 2, 3, 1},
                                                                 {1},
                                                                 {3},
                                                                 {1, 1, 1, 1},
                                                                 {1, 1, 1, 3},
                                                                 {1, 2},
                                                                 false,
                                                                 {1, 1, 1, 1},
                                                                 {1, 3, 1, 1},
                                                                 {1, 1, 1, 1},
                                                                 {1, 3, 1, 1},
                                                                 {2, 3},
                                                                 {0, 1}}));

struct TransposeReduceParams {
    // given params
    PartialShape transpose_input_shape;
    std::vector<int32_t> transpose_order;
    std::vector<int32_t> reduce_axes;
    bool reduction_keep_dims;

    // expected params
    std::vector<int32_t> ex_reduce_axes;
    std::vector<int32_t> ex_transpose_order;
};

class TransposeSinking : public ov::test::TestsCommon,
                         public testing::WithParamInterface<std::tuple<TransposeReduceParams, NodeTypeInfo>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& test_case = std::get<0>(GetParam());
        const auto& reduction_type_info = std::get<1>(GetParam());

        {
            auto input = std::make_shared<opset6::Parameter>(element::dynamic, test_case.transpose_input_shape);

            auto order = std::make_shared<opset6::Constant>(element::i64,
                                                            Shape{test_case.transpose_order.size()},
                                                            test_case.transpose_order);
            auto transpose = std::make_shared<opset6::Transpose>(input, order);

            auto axes = std::make_shared<opset6::Constant>(element::i64,
                                                           Shape{test_case.reduce_axes.size()},
                                                           test_case.reduce_axes);

            auto reduction = get_reduction(reduction_type_info, {transpose, axes}, test_case.reduction_keep_dims);

            f = std::make_shared<ov::Model>(NodeVector{reduction}, ParameterVector{input});
        }

        {
            auto input = std::make_shared<opset6::Parameter>(element::dynamic, test_case.transpose_input_shape);

            auto axes = std::make_shared<opset6::Constant>(element::i64,
                                                           Shape{test_case.ex_reduce_axes.size()},
                                                           test_case.ex_reduce_axes);
            auto reduction = get_reduction(reduction_type_info, {input, axes}, test_case.reduction_keep_dims);

            auto order = std::make_shared<opset6::Constant>(element::i64,
                                                            Shape{test_case.ex_transpose_order.size()},
                                                            test_case.ex_transpose_order);
            auto transpose = std::make_shared<opset6::Transpose>(reduction, order);

            f_ref = std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{input});
        }
    }

private:
    std::shared_ptr<Node> get_reduction(NodeTypeInfo reduction_type_info, const OutputVector& inputs, bool keep_dims) {
        std::shared_ptr<Node> reduction;
        for (const auto& it : get_available_opsets()) {
            const auto& opset = it.second();
            if (opset.contains_type(reduction_type_info)) {
                reduction = std::shared_ptr<Node>(opset.create(reduction_type_info.name));
                reduction->set_arguments(inputs);
                reduction->validate_and_infer_types();
            }
        }
        OPENVINO_ASSERT(reduction,
                        "supported opsets does not contain op with name: ",
                        reduction_type_info.name,
                        " version: ",
                        reduction_type_info.version_id);

        if (auto arithmetic_reduce = std::dynamic_pointer_cast<op::util::ArithmeticReductionKeepDims>(reduction))
            arithmetic_reduce->set_keep_dims(keep_dims);
        else if (auto logical_reduce = std::dynamic_pointer_cast<op::util::LogicalReductionKeepDims>(reduction))
            logical_reduce->set_keep_dims(keep_dims);
        reduction->validate_and_infer_types();
        return reduction;
    }
};

TEST_P(TransposeSinking, TransposeReduction) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::TransposeReduction>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);
    manager.run_passes(f);
    OV_ASSERT_NO_THROW(check_rt_info(f));

    auto fc = FunctionsComparator::no_default()
                  .enable(FunctionsComparator::NODES)
                  .enable(FunctionsComparator::PRECISIONS)
                  .enable(FunctionsComparator::CONST_VALUES);

    auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingReduces,
    TransposeSinking,
    testing::Combine(
        testing::Values(TransposeReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1, 2}, true, {2, 3}, {0, 2, 3, 1}},
                        TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70},
                                              {0, 6, 1, 5, 2, 4, 3},
                                              {1, 3, 6},
                                              true,
                                              {6, 5, 3},
                                              {0, 6, 1, 5, 2, 4, 3}},
                        TransposeReduceParams{{1, 3, 240, 140}, {0, 2, 3, 1}, {1, 2}, false, {2, 3}, {0, 1}},
                        TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70},
                                              {0, 6, 1, 5, 2, 4, 3},
                                              {1, 3, 6},
                                              false,
                                              {6, 5, 3},
                                              {0, 1, 2, 3}},
                        TransposeReduceParams{{10, 20, 30, 40, 50, 60, 70},
                                              {0, 6, 1, 5, 2, 4, 3},
                                              {1, -4, 6},
                                              false,
                                              {6, 5, 3},
                                              {0, 1, 2, 3}},
                        TransposeReduceParams{{1, 3, 240, 140}, {0, 1, 2, 3}, {0, 1, 2, -1}, false, {0, 1, 2, 3}, {}}),
        testing::Values(opset6::ReduceMax::get_type_info_static(),
                        opset6::ReduceMean::get_type_info_static(),
                        opset6::ReduceMin::get_type_info_static(),
                        opset6::ReduceProd::get_type_info_static(),
                        opset6::ReduceSum::get_type_info_static(),
                        opset6::ReduceL1::get_type_info_static(),
                        opset6::ReduceL2::get_type_info_static(),
                        opset6::ReduceLogicalAnd::get_type_info_static(),
                        opset6::ReduceLogicalOr::get_type_info_static())));

INSTANTIATE_TEST_SUITE_P(
    TransposeSinkingSqueeze,
    TransposeSinking,
    testing::Combine(testing::Values(TransposeReduceParams{{2, 3, 1, 1}, {0, 2, 3, 1}, {1, 2}, false, {2, 3}, {0, 1}},
                                     TransposeReduceParams{{10, 20, 30, 1, 50, 1, 1},
                                                           {0, 6, 1, 5, 2, 4, 3},
                                                           {1, 3, 6},
                                                           false,
                                                           {6, 5, 3},
                                                           {0, 1, 2, 3}}),
                     testing::Values(opset6::Squeeze::get_type_info_static())));

TEST_F(TransformationTestsF, TransposeFuseEliminatesTranspose) {
    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2});
        auto tr1_order = opset6::Constant::create(element::i64, Shape{5}, {0, 2, 3, 4, 1});
        auto transpose1 = std::make_shared<opset6::Transpose>(input, tr1_order);
        auto tr2_order = opset6::Constant::create(element::i64, Shape{5}, {0, 4, 1, 2, 3});
        auto transpose2 = std::make_shared<opset6::Transpose>(transpose1, tr2_order);
        auto add_const = opset6::Constant::create(element::f32, Shape{1}, {1});
        auto add = std::make_shared<opset6::Add>(transpose2, add_const);

        model = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input});
        manager.register_pass<ov::pass::TransposeFuse>();
    }

    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2});
        auto add_const = opset6::Constant::create(element::f32, Shape{1}, {1});
        auto add = std::make_shared<opset6::Add>(input, add_const);

        model_ref = std::make_shared<ov::Model>(NodeVector{add}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, TransposeFuses) {
    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2, 2});
        auto tr1_order = opset6::Constant::create(element::i64, Shape{6}, {0, 5, 1, 2, 3, 4});
        auto transpose1 = std::make_shared<opset6::Transpose>(input, tr1_order);
        auto tr2_order = opset6::Constant::create(element::i64, Shape{6}, {0, 1, 3, 4, 2, 5});
        auto transpose2 = std::make_shared<opset6::Transpose>(transpose1, tr2_order);
        auto result = std::make_shared<opset6::Result>(transpose2);
        result->set_layout("NC...");

        model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
        manager.register_pass<ov::pass::TransposeFuse>();
    }

    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2, 2});
        auto tr_order = opset6::Constant::create(element::i64, Shape{6}, {0, 5, 2, 3, 1, 4});
        auto transpose = std::make_shared<opset6::Transpose>(input, tr_order);
        auto result = std::make_shared<opset6::Result>(transpose);
        result->set_layout("NC...");

        model_ref = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, TransposeReduceNegative) {
    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 3, 64});
        auto order = opset6::Constant::create(element::i64, Shape{3}, {0, 2, 1});
        auto transpose = std::make_shared<opset6::Transpose>(input, order);
        auto axes = opset6::Constant::create(element::i64, Shape{}, {-1});
        auto reduce_mean = std::make_shared<opset6::ReduceMean>(transpose, axes, true);
        auto sub = std::make_shared<opset6::Subtract>(transpose, reduce_mean);

        model = std::make_shared<ov::Model>(NodeVector{sub}, ParameterVector{input});
        manager.register_pass<ov::pass::TransposeReduction>();
    }
}

TEST_F(TransformationTestsF, TransposeConvert) {
    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2, 2});
        auto order = opset6::Constant::create(element::i64, Shape{6}, {0, 5, 1, 2, 3, 4});
        auto transpose = std::make_shared<opset6::Transpose>(input, order);
        auto convert = std::make_shared<opset6::Convert>(transpose, element::f16);

        model = std::make_shared<ov::Model>(NodeVector{convert}, ParameterVector{input});
        manager.register_pass<ov::pass::TransposeConvert>();
    }

    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2, 2});
        auto convert = std::make_shared<opset6::Convert>(input, element::f16);
        auto order = opset6::Constant::create(element::i64, Shape{6}, {0, 5, 1, 2, 3, 4});
        auto transpose = std::make_shared<opset6::Transpose>(convert, order);

        model_ref = std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, TransposeConvertNegativeConsumers) {
    {
        auto input = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 640, 20, 2, 2});
        auto order = opset6::Constant::create(element::i64, Shape{6}, {0, 5, 1, 2, 3, 4});
        auto transpose = std::make_shared<opset6::Transpose>(input, order);
        auto convert = std::make_shared<opset6::Convert>(transpose, element::f16);

        model = std::make_shared<ov::Model>(NodeVector{convert, transpose}, ParameterVector{input});
        manager.register_pass<ov::pass::TransposeConvert>();
    }
}
