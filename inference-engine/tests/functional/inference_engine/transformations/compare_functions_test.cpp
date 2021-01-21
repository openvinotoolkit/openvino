// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <memory>
#include <queue>

#include <ngraph/pass/manager.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, CompareFunctoinsTIPositive) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        res_1->set_friendly_name("res_1");
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        res_2->set_friendly_name("res_2");
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompareFunctoinsTINegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto relu = std::make_shared<opset5::Relu>(lstm_cell);
        auto res_1 = std::make_shared<opset5::Result>(relu);
        res_1->set_friendly_name("res_1");
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        res_2->set_friendly_name("res_2");
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    EXPECT_FALSE(res.first);
    EXPECT_EQ(res.second, "LSTMCell/4 != Relu/0");
}