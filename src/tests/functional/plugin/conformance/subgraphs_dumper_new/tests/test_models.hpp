// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

inline std::shared_ptr<ov::Model> generate_abs_relu_add() {
    // param        param
    //   |            |
    //  abs          abs
    //   |            |
    // relu          relu
    //   |            |
    //    ------------
    //          |
    //         add
    //          |
    //        result
    std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
        std::make_shared<ov::op::v0::Abs>(test_parameter_0);
    std::shared_ptr<ov::op::v0::Relu> test_relu_0 =
        std::make_shared<ov::op::v0::Relu>(test_parameter_0);
    std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
        std::make_shared<ov::op::v0::Abs>(test_parameter_0);
    std::shared_ptr<ov::op::v0::Relu> test_relu_1 =
        std::make_shared<ov::op::v0::Relu>(test_abs_1);
    std::shared_ptr<ov::op::v1::Add> test_add_0 =
        std::make_shared<ov::op::v1::Add>(test_abs_0, test_abs_1);
    std::shared_ptr<ov::op::v0::Result> test_res =
        std::make_shared<ov::op::v0::Result>(test_add_0);
    return std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                       ov::ParameterVector{test_parameter_0, test_parameter_1});
}

inline std::shared_ptr<ov::Model> generate_abs_relu_abs_clamp_add() {
    // param        param              param        param
    //   |            |                  |            |
    //  abs          abs                abs          abs
    //   |            |                  |            |
    // relu         clamp               relu        clamp
    //   |            |                  |            |
    //    ------------                    ------------
    //          |                               |
    //         add                             add
    //          |                               |
    //           -------------------------------
    //                           |
    //                          add
    //                           |
    //                         result
    std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    std::shared_ptr<ov::op::v0::Abs> test_abs_0 =
        std::make_shared<ov::op::v0::Abs>(test_parameter_0);
    std::shared_ptr<ov::op::v0::Relu> test_relu_0 =
        std::make_shared<ov::op::v0::Relu>(test_abs_0);

    std::shared_ptr<ov::op::v0::Parameter> test_parameter_1 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
        std::make_shared<ov::op::v0::Abs>(test_parameter_1);
    std::shared_ptr<ov::op::v0::Clamp> test_clamp_1 =
        std::make_shared<ov::op::v0::Clamp>(test_abs_1, 0, 10);

    std::shared_ptr<ov::op::v1::Add> test_add_0 =
        std::make_shared<ov::op::v1::Add>(test_relu_0, test_clamp_1);

    std::shared_ptr<ov::op::v0::Parameter> test_parameter_0_1 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2});
    std::shared_ptr<ov::op::v0::Abs> test_abs_0_1 =
        std::make_shared<ov::op::v0::Abs>(test_parameter_0_1);
    std::shared_ptr<ov::op::v0::Relu> test_relu_0_1 =
        std::make_shared<ov::op::v0::Relu>(test_abs_0_1);

    std::shared_ptr<ov::op::v0::Parameter> test_parameter_1_1 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    std::shared_ptr<ov::op::v0::Abs> test_abs_1_1 =
        std::make_shared<ov::op::v0::Abs>(test_parameter_1_1);
    std::shared_ptr<ov::op::v0::Clamp> test_clamp_1_1 =
        std::make_shared<ov::op::v0::Clamp>(test_abs_1_1, 0, 10);

    std::shared_ptr<ov::op::v1::Add> test_add_0_1 =
        std::make_shared<ov::op::v1::Add>(test_relu_0_1, test_clamp_1_1);

    std::shared_ptr<ov::op::v1::Add> test_add_1 =
        std::make_shared<ov::op::v1::Add>(test_add_0, test_add_0_1);

    std::shared_ptr<ov::op::v0::Result> test_res =
        std::make_shared<ov::op::v0::Result>(test_add_1);
    return std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                       ov::ParameterVector{test_parameter_0, test_parameter_1,
                                       test_parameter_0_1, test_parameter_1_1});
}

inline std::shared_ptr<ov::Model> generate_abs_clamp_relu_abs_add() {
    // param
    //   |
    //  abs
    //   |
    // clamp
    //   |
    //  relu
    //   |
    //  abs         param
    //   |           |
    //   -------------
    //          |
    //         add
    //          |
    //        result
    std::shared_ptr<ov::op::v0::Parameter> test_parameter =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 5});
    std::shared_ptr<ov::op::v0::Parameter> test_parameter_0 =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 5});
    std::shared_ptr<ov::op::v0::Abs> test_abs =
        std::make_shared<ov::op::v0::Abs>(test_parameter_0);
    std::shared_ptr<ov::op::v0::Clamp> test_clamp =
        std::make_shared<ov::op::v0::Clamp>(test_abs, 0, 10);
    std::shared_ptr<ov::op::v0::Relu> test_relu =
        std::make_shared<ov::op::v0::Relu>(test_clamp);
    std::shared_ptr<ov::op::v0::Abs> test_abs_1 =
        std::make_shared<ov::op::v0::Abs>(test_relu);
    std::shared_ptr<ov::op::v1::Add> test_add =
        std::make_shared<ov::op::v1::Add>(test_abs_1, test_parameter_0);
    std::shared_ptr<ov::op::v0::Result> test_res =
        std::make_shared<ov::op::v0::Result>(test_add);
    return std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                       ov::ParameterVector{test_parameter, test_parameter_0});
}