// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <transformations/tensor_iterator_transformations/unroll_tensor_iterator.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/tensor_iterator_transformations/low_latency.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, LowLatencyLSTM) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512*16, 0);
        auto r_val = std::vector<float>(512*128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset4::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(lstm_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<opset4::TensorIterator::BodyLambda>(OutputVector{res_1, res_2},
                                                                         ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1, res_ti_2},
                                               ngraph::ParameterVector{X, Y, Z});

        auto file_ext = "svg";
        pass::VisualizeTree vt("low_latency_before" + std::string(".") + file_ext);
        std::vector<std::shared_ptr<Function>> functions = {f};
        vt.run_on_module(functions);

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        manager.register_pass<ngraph::pass::LSTMLowLatency>();
        manager.run_passes(f);

        pass::VisualizeTree n_vt("low_latency_after" + std::string(".") + file_ext);
        std::vector<std::shared_ptr<Function>> n_functions = {f};
        n_vt.run_on_module(n_functions);
        ASSERT_NO_THROW(check_rt_info(f));
    }
}

TEST(TransformationTests, LowLatencyGRU) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(384*16, 0);
        auto r_val = std::vector<float>(384*128, 0);
        auto b_val = std::vector<float>(384, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{384}, b_val);

        auto gru_cell = std::make_shared<opset4::GRUCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(gru_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(gru_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<opset4::TensorIterator::BodyLambda>(OutputVector{res_1, res_2},
                                                                         ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        //auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        manager.register_pass<ngraph::pass::GRULowLatency>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }
}

TEST(TransformationTests, LowLatencyRNN) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(128*16, 0);
        auto r_val = std::vector<float>(128*128, 0);
        auto b_val = std::vector<float>(128, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{128}, b_val);

        auto rnn_cell = std::make_shared<opset4::RNNCell>(squeeze, Yi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(rnn_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(rnn_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<opset4::TensorIterator::BodyLambda>(OutputVector{res_1, res_2},
                                                                         ParameterVector{Xi, Yi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        //auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        manager.register_pass<ngraph::pass::RNNLowLatency>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }
}

TEST(TransformationTests, LowLatencyLSTMReshape) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset4::Parameter>(element::f32, Shape{2, 1, 16});
        auto Y = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto axis = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto squeeze = std::make_shared<opset4::Squeeze>(Xi, axis);

        auto w_val = std::vector<float>(512*16, 0);
        auto r_val = std::vector<float>(512*128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset4::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset4::Result>(lstm_cell);
        auto unsqueeze = std::make_shared<opset4::Unsqueeze>(lstm_cell, axis);
        auto res_2 = std::make_shared<opset4::Result>(unsqueeze);
        auto body = std::make_shared<opset4::TensorIterator::BodyLambda>(OutputVector{res_1, res_2},
                                                                         ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset4::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 0);

        auto res_ti_1 = std::make_shared<opset4::Result>(tensor_iterator->output(1));
        auto res_ti_2 = std::make_shared<opset4::Result>(tensor_iterator->output(0));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1, res_ti_2},
                                               ngraph::ParameterVector{X, Y, Z});

        auto new_X = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 1, 16});
        f->replace_parameter(0, new_X);
        f->validate_nodes_and_infer_types();
        auto file_ext = "svg";
        pass::VisualizeTree vt("low_latency_before" + std::string(".") + file_ext);
        std::vector<std::shared_ptr<Function>> functions = {f};
        vt.run_on_module(functions);

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollTensorIterator>();
        manager.register_pass<ngraph::pass::LSTMLowLatency>();
        manager.run_passes(f);

        pass::VisualizeTree n_vt("low_latency_after" + std::string(".") + file_ext);
        std::vector<std::shared_ptr<Function>> n_functions = {f};
        n_vt.run_on_module(n_functions);
        ASSERT_NO_THROW(check_rt_info(f));
    }
}