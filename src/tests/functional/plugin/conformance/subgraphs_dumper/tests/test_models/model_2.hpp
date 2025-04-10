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

class Model_2 {
private:
    using PatternBorders = ov::tools::subgraph_dumper::RepeatPatternExtractor::PatternBorders;

public:
    Model_2() {
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
            std::make_shared<ov::op::v0::Abs>(test_parameter);
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
        model =  std::make_shared<ov::Model>(ov::ResultVector{test_res},
                                             ov::ParameterVector{test_parameter, test_parameter_0});
    }

    std::shared_ptr<ov::Model> get() {
        return model;
    }

    std::vector<std::shared_ptr<ov::Model>> get_repeat_pattern_ref() {
        return std::vector<std::shared_ptr<ov::Model>>();
    }

    std::vector<std::vector<ov::NodeVector>>
    get_ref_node_vector() { return ref_nodes; }

    std::vector<std::vector<PatternBorders>>
    get_ref_node_borders() { return ref_borders; }

protected:
    std::shared_ptr<ov::Model> model;
    std::vector<std::vector<ov::NodeVector>> ref_nodes;
    std::vector<std::vector<PatternBorders>> ref_borders;
};
