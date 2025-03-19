// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov;

//
// Graph before (params in [] brackets, constants in () parens, results in {} braces):
//
//  [x]   [y]   [z]
//    \   /      |
//     Add   (k) |
//       \   /   |
//        Mul**  |
//          \   /
//           Sub
//            |
//           {r}
//
// Param substitutions:
//
//    [x] -> [x']
//
// Body substitutions:
//
//    (k) -> (k')
//    [y] -> (k'')
//    [z] -> [x'] + **
//
// After replacement:
//
//  [x']---------
//    |          |
//    |   (k'')  |    [z] and [y] is still there, but dead
//    \   /      |
//     Add  (k') |
//       \   /   |
//        Mul    |
//          \   /
//           Sub ***
//            |
//           {r}
//
TEST(replace_node, replace_nodes) {
    auto x = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto z = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});

    auto add = make_shared<op::v1::Add>(x, y);
    auto k = make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, vector<float>{1, 2});
    auto mul = make_shared<op::v1::Multiply>(add, k);
    auto sub = make_shared<op::v1::Subtract>(mul, z);

    auto f = make_shared<Model>(NodeVector{sub}, ParameterVector{x, y, z});

    unordered_map<shared_ptr<ov::op::v0::Parameter>, shared_ptr<ov::op::v0::Parameter>> parameter_replacement_map;
    auto x_replacement = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    parameter_replacement_map[x] = x_replacement;

    unordered_map<shared_ptr<Node>, shared_ptr<Node>> body_replacement_map;
    auto y_replacement = make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, vector<float>{3, 4});
    auto k_replacement = make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, vector<float>{5, 6});
    auto z_replacement = make_shared<op::v1::Add>(x_replacement, mul);
    body_replacement_map[y] = y_replacement;
    body_replacement_map[k] = k_replacement;
    body_replacement_map[z] = z_replacement;

    replace_nodes(f, parameter_replacement_map, body_replacement_map);

    // Should still have three params.
    ASSERT_EQ(f->get_parameters().size(), 3);

    // The three params be {x_replacement, y, z}.
    ASSERT_EQ(f->get_parameters()[0], x_replacement);
    ASSERT_EQ(f->get_parameters()[1], y);
    ASSERT_EQ(f->get_parameters()[2], z);

    // y, z should be dead.
    ASSERT_EQ(y->get_users(true).size(), 0);
    ASSERT_EQ(z->get_users(true).size(), 0);

    // Should still have one result.
    ASSERT_EQ(f->get_results().size(), 1);

    // Result node should be sub (unchanged).
    ASSERT_EQ(f->get_results()[0]->get_input_node_shared_ptr(0), sub);

    // sub's arguments should be mul (unchanged) and z_replacement.
    ASSERT_EQ(sub->get_input_node_shared_ptr(0), mul);
    ASSERT_EQ(sub->get_input_node_shared_ptr(1), z_replacement);

    // mul's arguments should be add (unchanged) and k_replacement.
    ASSERT_EQ(mul->get_input_node_shared_ptr(0), add);
    ASSERT_EQ(mul->get_input_node_shared_ptr(1), k_replacement);

    // add's arguments should be x_replacement and y_replacement.
    ASSERT_EQ(add->get_input_node_shared_ptr(0), x_replacement);
    ASSERT_EQ(add->get_input_node_shared_ptr(1), y_replacement);

    // z_replacement's arguments should be x_replacement and mul.
    ASSERT_EQ(z_replacement->get_input_node_shared_ptr(0), x_replacement);
    ASSERT_EQ(z_replacement->get_input_node_shared_ptr(1), mul);
}

TEST(replace_node, simple_node_replacement) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 64});
    param->output(0).get_tensor().set_names({"a", "b"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    relu->output(0).get_tensor().set_names({"c", "d"});

    auto new_relu = std::make_shared<ov::op::v0::Relu>(param);
    new_relu->output(0).get_tensor().set_names({"f"});
    replace_node(relu, new_relu);

    ASSERT_EQ(new_relu->output(0).get_tensor().get_names(), std::unordered_set<std::string>({"c", "d", "f"}));
}

TEST(replace_node, replacement_with_direct_parent_node) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 64});
    param->output(0).get_tensor().set_names({"a", "b"});

    auto child_1 = std::make_shared<ov::op::v0::Relu>(param);
    auto child_2 = std::make_shared<ov::op::v0::Relu>(param);

    auto model = std::make_shared<ov::Model>(OutputVector{child_1, child_2}, ParameterVector{param});
    OV_ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    relu->output(0).get_tensor().set_names({"c", "d"});
    replace_node(param, relu);

    // This check validates that the model is consistent and contains no loops.
    // The topological sorting throws an exception in case of a loop in the graph.
    OV_ASSERT_NO_THROW(model->validate_nodes_and_infer_types());

    int relu_cnt = 0;
    for (const auto& op : model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v0::Relu>(op)) {
            relu_cnt++;
        }
    }
    ASSERT_EQ(relu_cnt, 3);
}

TEST(replace_node, node_elimination) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 64});
    param->output(0).get_tensor().set_names({"a", "b"});
    auto relu1 = std::make_shared<ov::op::v0::Relu>(param);
    relu1->output(0).get_tensor().set_names({"c", "d"});
    auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);
    relu2->output(0).get_tensor().set_names({"e", "f"});

    ASSERT_TRUE(replace_output_update_name(relu2->output(0), relu2->input_value(0)));
    ASSERT_EQ(relu1->output(0).get_tensor().get_names(), std::unordered_set<std::string>({"c", "d", "e", "f"}));
    ASSERT_EQ(param->output(0).get_tensor().get_names(), std::unordered_set<std::string>({"a", "b"}));
}

TEST(replace_node, node_elimination_1) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 64});
    auto split = std::make_shared<op::v1::Split>(param, ov::op::v0::Constant::create(element::i64, Shape{}, {0}), 3);
    auto relu1 = std::make_shared<ov::op::v0::Relu>(split->output(2));
    auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);
    auto result2 = std::make_shared<ov::op::v0::Result>(relu2);

    // relu1 can be removed because we don't have to preserve name
    ASSERT_TRUE(replace_output_update_name(relu1->output(0), relu1->input_value(0)));

    // relu2 can't be removed because we have to preserve name and Split has more than one output port
    ASSERT_FALSE(replace_output_update_name(relu2->output(0), relu2->input_value(0)));
}

TEST(replace_node, node_elimination_2) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 64});
    auto relu1 = std::make_shared<ov::op::v0::Relu>(param);
    auto result1 = std::make_shared<ov::op::v0::Result>(relu1);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);
    auto result2 = std::make_shared<ov::op::v0::Result>(relu2);

    // relu2 can't be removed because relu1 has Result as consumer
    ASSERT_FALSE(replace_output_update_name(relu2->output(0), relu2->input_value(0)));
}

TEST(replace_node, node_elimination_3) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 64});
    auto relu1 = std::make_shared<ov::op::v0::Relu>(param);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(relu1);
    auto relu3 = std::make_shared<ov::op::v0::Relu>(relu1);
    auto result2 = std::make_shared<ov::op::v0::Result>(relu3);

    // relu3 can be removed because relu1 has no Result as consumer
    ASSERT_TRUE(replace_output_update_name(relu3->output(0), relu3->input_value(0)));
}

TEST(replace_node, node_elimination_4) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 64});
    auto relu1 = std::make_shared<ov::op::v0::Relu>(param);
    auto split = std::make_shared<op::v1::Split>(relu1, ov::op::v0::Constant::create(element::i64, Shape{}, {0}), 3);
    auto relu2 = std::make_shared<ov::op::v0::Relu>(split->output(2));
    auto result2 = std::make_shared<ov::op::v0::Result>(relu2);

    ASSERT_TRUE(replace_output_update_name(split->output(2), split->input_value(0)));
}

TEST(replace_node, output_replacement) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 64});
    param->output(0).get_tensor().set_names({"a", "b"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    relu->output(0).get_tensor().set_names({"c", "d"});

    auto new_relu = std::make_shared<ov::op::v0::Relu>(param);
    new_relu->output(0).get_tensor().set_names({"f"});

    relu->output(0).replace(new_relu->output(0));

    ASSERT_EQ(new_relu->output(0).get_tensor().get_names(), std::unordered_set<std::string>({"c", "d", "f"}));
}

TEST(replace_node, source_replacement) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 64});
    param->output(0).get_tensor().set_names({"a", "b"});

    auto param1 = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1, 64});
    param1->output(0).get_tensor().set_names({"c", "d"});

    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    relu->input(0).replace_source_output(param1->output(0));

    ASSERT_EQ(param->output(0).get_tensor().get_names(), std::unordered_set<std::string>({"a", "b"}));
    ASSERT_EQ(param1->output(0).get_tensor().get_names(), std::unordered_set<std::string>({"c", "d"}));
}
