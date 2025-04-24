// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"

class Model_1 {
private:
    using PatternBorders = ov::tools::subgraph_dumper::RepeatPatternExtractor::PatternBorders;

public:
    Model_1() {
        // param        param              param        param
        //   |            |                  |            |
        //  abs          abs                abs          abs
        //   |            |                  |            |
        // relu         clamp               relu        clamp
        //   |            |                  |            |
        //    ------------                    ------------
        //          |                               |
        //         add                           Multiply                 param          param
        //          |                               |                    |              |
        //           -------------------------------                      --------------
        //                           |                                           |
        //                        Multiply                                   multiply
        //                           |                                           |
        //                          Relu                                        Relu
        //                           |                                           |
        //                            -------------------------------------------
        //                                                  |
        //                                               subtract
        //                                                  |
        //                                                result
        size_t op_idx = 0;

        std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
        test_parameter_0->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
            std::make_shared<ov::op::v0::Abs>(test_parameter_0);
        test_abs_0->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Relu> test_relu_0 =
            std::make_shared<ov::op::v0::Relu>(test_abs_0);
        test_relu_0->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
        test_parameter_1->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
            std::make_shared<ov::op::v0::Abs>(test_parameter_1);
        test_abs_1->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Clamp> test_clamp_1 =
            std::make_shared<ov::op::v0::Clamp>(test_abs_1, 0, 10);
        test_clamp_1->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v1::Add> test_add_0 =
            std::make_shared<ov::op::v1::Add>(test_relu_0, test_clamp_1);
        test_add_0->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Parameter> test_parameter_0_0 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
        test_parameter_0_0->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Abs> test_abs_0_0 =
            std::make_shared<ov::op::v0::Abs>(test_parameter_0_0);
        test_abs_0_0->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Relu> test_relu_0_0 =
            std::make_shared<ov::op::v0::Relu>(test_abs_0_0);
        test_relu_0_0->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Parameter> test_parameter_0_1 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
        test_parameter_0_1->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Abs> test_abs_0_1 =
            std::make_shared<ov::op::v0::Abs>(test_parameter_0_1);
        test_abs_0_1->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Clamp> test_clamp_0_1 =
            std::make_shared<ov::op::v0::Clamp>(test_abs_0_1, 0, 10);
        test_clamp_0_1->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v1::Multiply> test_multiply_0_0 =
            std::make_shared<ov::op::v1::Multiply>(test_relu_0_0, test_clamp_0_1);
        test_multiply_0_0->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v1::Multiply> test_multiply_0_1 =
            std::make_shared<ov::op::v1::Multiply>(test_add_0, test_multiply_0_0);
        test_multiply_0_1->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Relu> test_relu_0_1 =
            std::make_shared<ov::op::v0::Relu>(test_multiply_0_1);
        test_relu_0_1->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Parameter> test_parameter_1_0 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
        test_parameter_1_0->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Parameter> test_parameter_1_1 =
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
        test_parameter_1_1->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v1::Multiply> test_multiply_1_1 =
            std::make_shared<ov::op::v1::Multiply>(test_parameter_1_0, test_parameter_1_1);
        test_multiply_1_1->set_friendly_name("Op_" + std::to_string(op_idx++));
        std::shared_ptr<ov::op::v0::Relu> test_relu_1_1 =
            std::make_shared<ov::op::v0::Relu>(test_multiply_1_1);
        test_relu_1_1->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v1::Subtract> test_add =
            std::make_shared<ov::op::v1::Subtract>(test_relu_0_1, test_relu_1_1);
        test_add->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Result> test_res =
            std::make_shared<ov::op::v0::Result>(test_add);
        test_res->set_friendly_name("Op_" + std::to_string(op_idx++));
        model = std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                            ov::ParameterVector{test_parameter_0, test_parameter_1,
                                                                test_parameter_0_0, test_parameter_0_1,
                                                                test_parameter_1_0, test_parameter_1_1});

        ref_nodes = {{{test_abs_0, test_relu_0}, {test_abs_0_0, test_relu_0_0}},
                     {{test_abs_1, test_clamp_1}, {test_abs_0_1, test_clamp_0_1}},
                     {{test_multiply_0_1, test_relu_0_1}, {test_multiply_1_1, test_relu_1_1}}};
        {
            PatternBorders ref_pattern_0 = {test_abs_0->inputs(), test_relu_0->outputs()},
                           ref_pattern_0_0 = {test_abs_0_0->inputs(), test_relu_0_0->outputs()},
                           ref_pattern_1 = {test_abs_1->inputs(), test_clamp_1->outputs()},
                           ref_pattern_0_1_0 = {test_abs_0_1->inputs(), test_clamp_0_1->outputs()},
                           test_pattern_0_1_1 = {test_multiply_0_1->inputs(), test_relu_0_1->outputs()},
                           test_pattern_1_1 = {test_multiply_1_1->inputs(), test_relu_1_1->outputs()};
            std::vector<std::vector<PatternBorders>> ref_res = {{ref_pattern_0_0, ref_pattern_0},
                                                                {ref_pattern_0_1_0, ref_pattern_1},
                                                                {test_pattern_1_1, test_pattern_0_1_1}};
            ref_borders = std::move(ref_res);
        }
        start_ops = {test_abs_0, test_abs_0_0, test_abs_0_1, test_abs_1};
        out_nodes = {test_abs_0, test_relu_0, test_add_0, test_multiply_0_1,
                     test_relu_0_1, test_add};
        start_node = test_abs_0;
    }

    std::shared_ptr<ov::Model> get() {
        return model;
    }

    std::vector<std::shared_ptr<ov::Model>> get_repeat_pattern_ref() {
        std::vector<std::shared_ptr<ov::Model>> ref;
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
                std::make_shared<ov::op::v0::Abs>(test_parameter_0);
            std::shared_ptr<ov::op::v0::Relu> test_relu_0 =
                std::make_shared<ov::op::v0::Relu>(test_abs_0);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_relu_0);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_0});
            ref.push_back(ref_model);
        }
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
                std::make_shared<ov::op::v0::Abs>(test_parameter_0);
            std::shared_ptr<ov::op::v0::Clamp> test_clamp_1 =
                std::make_shared<ov::op::v0::Clamp>(test_abs_1, 0, 10);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_clamp_1);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_0});
            ref.push_back(ref_model);
        }
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_1_0 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_1_1 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v1::Multiply> test_multiply_1 =
                std::make_shared<ov::op::v1::Multiply>(test_parameter_1_0, test_parameter_1_1);
            std::shared_ptr<ov::op::v0::Relu> test_relu_1 =
                std::make_shared<ov::op::v0::Relu>(test_multiply_1);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_relu_1);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_1_0, test_parameter_1_1});
            ref.push_back(ref_model);
        }
        return ref;
    }

    std::vector<std::vector<ov::NodeVector>>
    get_ref_node_vector() { return ref_nodes; }

    std::vector<std::vector<PatternBorders>>
    get_ref_node_borders() { return ref_borders; }

    ov::NodeVector
    get_start_ops() { return start_ops; }

    std::unordered_set<std::shared_ptr<ov::Node>>
    get_out_nodes_after_abs_0() {
        return out_nodes;
    }

    std::shared_ptr<ov::Node>
    get_test_abs_0() {
        return start_node;
    }

protected:
    std::shared_ptr<ov::Model> model;
    std::vector<std::vector<ov::NodeVector>> ref_nodes;
    std::vector<std::vector<PatternBorders>> ref_borders;
    ov::NodeVector start_ops;
    std::unordered_set<std::shared_ptr<ov::Node>> out_nodes;
    std::shared_ptr<ov::Node> start_node;
};
