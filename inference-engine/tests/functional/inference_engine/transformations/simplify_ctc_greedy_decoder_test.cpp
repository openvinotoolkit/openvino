// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>


#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/simplify_ctc_greedy_decoder.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, SimplifyCTCGreedyDecoderTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 7 });
        auto seq_len = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{ 1 });

        auto decoder_v6 = std::make_shared<ngraph::op::v6::CTCGreedyDecoderSeqLen>(data, seq_len, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ decoder_v6 }, ngraph::ParameterVector{ data, seq_len });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SimplifyCTCGreedyDecoder>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 3 });
        auto seq_len1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{ 1 });

        auto data_shape = std::make_shared<ngraph::opset6::ShapeOf>(data1);
        auto constT_0 = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {-1});
        auto constT_1 = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto T = std::make_shared<ngraph::opset6::Gather>(data_shape, constT_1, constT_0);

        auto constN_0 = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto constN_1 = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {-1});
        auto N = std::make_shared<ngraph::opset6::Gather>(data_shape, constN_0, constN_0);

        auto start = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t >({1}));
        auto step = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, std::vector<int64_t >({1}));
        auto range1T = std::make_shared<ngraph::opset6::Range>(start, T, step,
                                                               ngraph::element::i64);

        auto constUnsqueeze1 = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto tT = std::make_shared<ngraph::opset6::Unsqueeze>(T, constUnsqueeze1);
        auto constUnsqueeze2 = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto tN = std::make_shared<ngraph::opset6::Unsqueeze>(N, constUnsqueeze2);
        auto mask_shape = std::make_shared<ngraph::opset6::Concat>(
                ngraph::OutputVector{tT->output(0), tN->output(0)}, 0);
        auto upper_bounds = std::make_shared<ngraph::opset6::Broadcast>(
                seq_len1, mask_shape->output(0));
        auto bool_seq_mask = std::make_shared<ngraph::opset6::GreaterEqual>(upper_bounds->output(0),
                                                                            range1T->output(0));
        auto const_0f = ngraph::opset6::Constant::create(ngraph::element::f64, ngraph::Shape{}, {0.0});
        auto const_1f = ngraph::opset6::Constant::create(ngraph::element::f64, ngraph::Shape{}, {1.0});
        auto seq_mask = std::make_shared<ngraph::opset6::Select>(bool_seq_mask, const_1f, const_0f);

        auto decoder_v1 = std::make_shared<ngraph::opset6::CTCGreedyDecoder>(data1, seq_mask, true);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ decoder_v1 }, ngraph::ParameterVector{ data1, seq_len1 });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}
