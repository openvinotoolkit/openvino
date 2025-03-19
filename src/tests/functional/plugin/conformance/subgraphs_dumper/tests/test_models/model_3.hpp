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
#include "openvino/op/concat.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/split.hpp"
#include "matchers/subgraph/repeat_pattern.hpp"

class Model_3 {
protected:
    using PatternBorders = ov::tools::subgraph_dumper::RepeatPatternExtractor::PatternBorders;
    std::shared_ptr<ov::Model> model;
    ov::NodeVector start_ops;
    ov::NodeVector node_queue;
    std::vector<std::vector<std::pair<std::shared_ptr<ov::Node>, std::vector<size_t>>>> ordered_patterns;
    std::vector<ov::NodeVector> repeats;

public:
    Model_3() {
        //                        param_00
        //                           |
        //                         relu_0
        //                           |
        //                        split_1
        //                           |
        //                    +-------------+
        //                    |             |
        //                 relu_2         clamp_3
        //                    |             |
        //                split_4           |
        //                    |             |
        //             +------------+       |
        //             |            |       |
        //         relu_5         clamp_6   |
        //             |            |       |
        //             +------------+       |
        //                    |             |
        //                 add_7            |
        //                    |             |
        //                 concat_8         |
        //                    |             |
        //                    +-------------+
        //                            |
        //                         multiply_9          param_01
        //                            |------------------+
        //                           add_10            param_02
        //                            |------------------+
        //                         multiply_11
        //                            |
        //                          result_00

        auto param_00 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 24, 1});
        param_00->set_friendly_name("param_00");
        auto relu_0 = std::make_shared<ov::op::v0::Relu>(param_00);
        relu_0->set_friendly_name("relu_0");
        auto axis_split_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>({2}));
        auto split_1 = std::make_shared<ov::op::v1::Split>(relu_0, axis_split_1, 2);
        split_1->set_friendly_name("split_1");
        auto relu_2 = std::make_shared<ov::op::v0::Relu>(split_1->output(0));
        relu_2->set_friendly_name("relu_2");
        auto clamp_3 = std::make_shared<ov::op::v0::Clamp>(split_1->output(1), 0 , 10);
        clamp_3->set_friendly_name("clamp_3");
        auto axis_split_4 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>({2}));
        auto split_4 = std::make_shared<ov::op::v1::Split>(relu_2, axis_split_4, 2);
        split_4->set_friendly_name("split_4");
        auto relu_5 = std::make_shared<ov::op::v0::Relu>(split_4->output(0));
        relu_5->set_friendly_name("relu_5");
        auto clamp_6 = std::make_shared<ov::op::v0::Clamp>(split_4->output(1), 0, 10);
        clamp_6->set_friendly_name("clamp_6");
        auto add_7 = std::make_shared<ov::op::v1::Add>(relu_5, clamp_6);
        add_7->set_friendly_name("add_7");
        auto param_03 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, add_7->get_shape());
        param_03->set_friendly_name("param_03");
        auto concat_8 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{add_7, param_03}, 2);
        concat_8->set_friendly_name("concat_8");
        auto multiply_9 = std::make_shared<ov::op::v1::Multiply>(concat_8, clamp_3);
        multiply_9->set_friendly_name("multiply_9");
        auto param_01 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, multiply_9->get_shape());
        param_01->set_friendly_name("param_01");
        auto add_10 = std::make_shared<ov::op::v1::Add>(multiply_9, param_01);
        add_10->set_friendly_name("add_10");
        auto param_02 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, add_10->get_shape());
        param_02->set_friendly_name("param_02");
        auto multiply_11 = std::make_shared<ov::op::v1::Multiply>(add_10, param_02);
        multiply_11->set_friendly_name("multiply_11");
        auto result_00 = std::make_shared<ov::op::v0::Result>(multiply_11);
        result_00->set_friendly_name("result_00");

        model = std::make_shared<ov::Model>(ov::ResultVector{result_00},
                                            ov::ParameterVector{param_00, param_01, param_02, param_03});

        start_ops = {relu_0, relu_2, relu_5};
        ordered_patterns = {{
            { relu_0, {}},
            { split_1, {0}},
            { relu_2, {1}},
            { split_4, {2}},
            { relu_5, {3}},
            { clamp_6, {3}},
            { add_7, {4, 5}},
            { concat_8, {6}},
            { clamp_3, {1}},
            { multiply_9, {7, 8}},
            { add_10, {9}},
            { multiply_11, {10}},
        }, {
            { relu_2, {}},
            { split_4, {0}},
            { relu_5, {1}},
            { clamp_6, {1}},
            { add_7, {2, 3}},
            { concat_8, {4}},
            { multiply_9, {5}},
            { add_10, {6}},
            { multiply_11, {7}},
        }, {
            { relu_5, {}},
            { add_7, {0}},
            { concat_8, {1}},
            { multiply_9, {2}},
            { add_10, {3}},
            { multiply_11, {4}},
        }};
        repeats = {
            { relu_0, split_1, relu_2 },
            { relu_2, split_4, relu_5 },
        };
        node_queue = {
            relu_0, split_1, relu_2, split_4, relu_5, clamp_6,
            add_7, concat_8, clamp_3, multiply_9, add_10, multiply_11,
        };
    }

    std::shared_ptr<ov::Model> get() {
        return model;
    }

    ov::NodeVector
    get_start_ops() { return start_ops; }

    std::vector<std::vector<std::pair<std::shared_ptr<ov::Node>, std::vector<size_t>>>>
    get_ordered_patterns() { return ordered_patterns; }

    std::vector<ov::NodeVector>
    get_repeats() { return repeats; }

    ov::NodeVector
    get_queue() { return node_queue; }
};
