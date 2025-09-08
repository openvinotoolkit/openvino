// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"

using namespace std;
using namespace ov;

TEST(input_output, param_tensor) {
    // Params have no arguments, so we can check that the value becomes a tensor output
    auto& et = ov::element::f32;
    ov::Shape shape{2, 4};
    auto param = make_shared<ov::op::v0::Parameter>(et, shape);

    ASSERT_EQ(param->get_output_size(), 1);
    ASSERT_EQ(et, param->get_element_type());
    ASSERT_EQ(shape, param->get_shape());
}

TEST(input_output, simple_output) {
    auto param_0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto param_1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto add = make_shared<op::v1::Add>(param_0, param_1);

    // Sort the ops
    vector<shared_ptr<Node>> nodes;
    nodes.push_back(param_0);
    nodes.push_back(param_1);
    nodes.push_back(add);

    // At this point, the add should have each input associated with the output of the appropriate
    // parameter
    ASSERT_EQ(1, add->get_output_size());
    ASSERT_EQ(2, add->get_input_size());
    for (size_t i = 0; i < add->get_input_size(); i++) {
        ASSERT_EQ(add->input_value(i).get_node_shared_ptr(), nodes.at(i));
    }
}
