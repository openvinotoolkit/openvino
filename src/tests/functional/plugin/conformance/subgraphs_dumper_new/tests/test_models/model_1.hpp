// Copyright (C) 2018-2023 Intel Corporation
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

class Model_1 {
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
        //         add                             add                 param          param
        //          |                               |                    |              |
        //           -------------------------------                      --------------
        //                           |                                           |
        //                        Multiply                                   multiply
        //                           |                                           |
        //                          Relu                                        Relu
        //                           |                                           |
        //                            -------------------------------------------
        //                                                  |
        //                                               Multiply
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

        std::shared_ptr<ov::op::v1::Add> test_add_0_0 =
            std::make_shared<ov::op::v1::Add>(test_relu_0_0, test_clamp_0_1);
        test_add_0_0->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v1::Multiply> test_multiply_0_0 =
            std::make_shared<ov::op::v1::Multiply>(test_add_0, test_add_0_0);
        test_multiply_0_0->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Relu> test_relu_0_1 =
            std::make_shared<ov::op::v0::Relu>(test_multiply_0_0);
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

        std::shared_ptr<ov::op::v1::Add> test_add =
            std::make_shared<ov::op::v1::Add>(test_relu_0_1, test_relu_1_1);
        test_add->set_friendly_name("Op_" + std::to_string(op_idx++));

        std::shared_ptr<ov::op::v0::Result> test_res =
            std::make_shared<ov::op::v0::Result>(test_add);
        test_res->set_friendly_name("Op_" + std::to_string(op_idx++));
        model = std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                            ov::ParameterVector{test_parameter_0, test_parameter_1,
                                                                test_parameter_0_0, test_parameter_0_1,
                                                                test_parameter_1_0, test_parameter_1_1});
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
            std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
                std::make_shared<ov::op::v0::Abs>(test_parameter_0);
            std::shared_ptr<ov::op::v0::Relu> test_relu_0 =
                std::make_shared<ov::op::v0::Relu>(test_abs_0);
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1});
            std::shared_ptr<ov::op::v1::Add> test_add =
                std::make_shared<ov::op::v1::Add>(test_relu_0, test_parameter_1);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_add);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_0, test_parameter_1});
            ref.push_back(ref_model);
        }
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
                std::make_shared<ov::op::v0::Abs>(test_parameter_1);
            std::shared_ptr<ov::op::v0::Clamp> test_clamp_1 =
                std::make_shared<ov::op::v0::Clamp>(test_abs_1, 0, 10);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_clamp_1);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_0, test_parameter_1});
            ref.push_back(ref_model);
        }
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
                std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
                std::make_shared<ov::op::v0::Abs>(test_parameter_1);
            std::shared_ptr<ov::op::v0::Clamp> test_clamp_1 =
                std::make_shared<ov::op::v0::Clamp>(test_abs_1, 0, 10);
            std::shared_ptr<ov::op::v1::Add> test_add =
                std::make_shared<ov::op::v1::Add>(test_parameter_0, test_clamp_1);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_add);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_0, test_parameter_1});
            ref.push_back(ref_model);
        }
        {
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
            std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
                std::make_shared<ov::op::v0::Abs>(test_parameter_1);
            std::shared_ptr<ov::op::v0::Clamp> test_clamp_1 =
                std::make_shared<ov::op::v0::Clamp>(test_abs_1, 0, 10);
            std::shared_ptr<ov::op::v1::Add> test_add =
                std::make_shared<ov::op::v1::Add>(test_parameter_0, test_clamp_1);
            std::shared_ptr<ov::op::v0::Result> res =
                std::make_shared<ov::op::v0::Result>(test_add);
            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{res},
                                                         ov::ParameterVector{test_parameter_0, test_parameter_1});
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
        }
        return ref;
    }

protected:
    std::shared_ptr<ov::Model> model;
};
