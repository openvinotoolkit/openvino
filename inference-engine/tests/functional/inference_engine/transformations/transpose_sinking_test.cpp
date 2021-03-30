// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/transpose_sinking_fq_reduce.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

struct Transpose_FQ_Reduce_params {
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

class TransposeSinking : public CommonTestUtils::TestsCommon,
                         public testing::WithParamInterface<std::tuple<Transpose_FQ_Reduce_params>> {
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

TEST_P(TransposeSinking, TransposeFQReduce) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::TransposeSinkingFQReduce>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    auto res = compare_functions(f, f_ref, true);
    ASSERT_TRUE(res.first) << res.second;
}


INSTANTIATE_TEST_CASE_P(TransformationTest, TransposeSinking, testing::Values(
        Transpose_FQ_Reduce_params{{1, 3, 240, 140}, {0, 2, 3, 1}, {1}, {3}, {1, 1, 1, 1}, {1, 1, 1, 3}, {1, 2}, true,
                                   {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {2, 3}, {0, 2, 3, 1}},
        Transpose_FQ_Reduce_params{{1, 3, 240, 140}, {0, 2, 3, 1}, {1}, {3}, {1, 1, 1, 1}, {1, 1, 1, 3}, {1, 2}, false,
                                   {1, 1, 1, 1}, {1, 3, 1, 1}, {1, 1, 1, 1}, {1, 3, 1, 1}, {2, 3}, {0, 1}}));
