// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/node_builders/broadcast.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/variable.hpp"

using namespace std;
using namespace ov;

TEST(build_graph, build_simple) {
    // Function with 4 parameters
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 3});
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 7});
    auto arg3 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 7});
    auto broadcast_1 = ov::test::utils::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = ov::test::utils::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<op::v0::MatMul>(arg2, arg0);
    ASSERT_EQ(dot->input_value(0).get_node_shared_ptr(), arg2);
    ASSERT_EQ(dot->input_value(1).get_node_shared_ptr(), arg0);

    auto cluster_0 = make_shared<Model>(dot, ParameterVector{arg0, arg1, arg2, arg3});

    ASSERT_EQ(cluster_0->get_output_op(0)->input_value(0).get_node_shared_ptr(), dot);
}

TEST(build_graph, literal) {
    // float scalar from a float
    // auto float0 = FloatConstant::make(3.0);
    vector<float> float_t{3.0};
    auto float0 = make_shared<op::v0::Constant>(element::f32, Shape{1}, float_t);
    ASSERT_EQ(float0->get_vector<float>(), std::vector<float>{3.0});
    ASSERT_EQ(float0->get_element_type(), element::f32);
    ASSERT_EQ(float0->get_shape(), Shape{1});
    auto d = make_shared<op::v0::MatMul>(float0, float0);
    ASSERT_EQ(d->input_values().at(0).get_node_shared_ptr(), float0);
    ASSERT_EQ(d->input_values().at(1).get_node_shared_ptr(), float0);

    vector<int32_t> int32{3};
    auto int32_0 = make_shared<op::v0::Constant>(element::i32, Shape{}, int32);
    ASSERT_EQ(int32_0->get_vector<int32_t>(), std::vector<int>{3});
    ASSERT_EQ(int32_0->get_element_type(), element::i32);
    ASSERT_EQ(int32_0->get_shape(), Shape{});
}

TEST(build_graph, tensor) {
    // float scalar from a float
    // auto float0 = FloatConstant::make(3.0);
    Shape shape{2, 3};
    vector<float> float_t(shape_size(shape), 0);
    auto float0 = make_shared<op::v0::Constant>(element::f32, shape, float_t);
    ASSERT_EQ(float0->get_element_type(), element::f32);
    ASSERT_EQ(float0->get_shape(), shape);
    auto d = make_shared<op::v1::Add>(float0, float0);
    ASSERT_EQ(d->input_values().at(0).get_node_shared_ptr(), float0);
    ASSERT_EQ(d->input_values().at(1).get_node_shared_ptr(), float0);

    Shape ishape{3, 5};
    vector<int32_t> idata(shape_size(ishape), 0);
    auto int32_0 = make_shared<op::v0::Constant>(element::i32, ishape, idata);
    ASSERT_EQ(int32_0->get_element_type(), element::i32);
    ASSERT_EQ(int32_0->get_shape(), ishape);
}

// Check functions with undeclared parameters
TEST(build_graph, function_undeclared_parameters) {
    // Function with 4 parameters
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7, 3});
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 7});
    auto arg3 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 7});
    auto broadcast_1 = ov::test::utils::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = ov::test::utils::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<op::v0::MatMul>(arg2, arg0);
    ASSERT_EQ(dot->input_values()[0].get_node_shared_ptr(), arg2);
    ASSERT_EQ(dot->input_values()[1].get_node_shared_ptr(), arg0);
    try {
        auto f = make_shared<Model>(dot, ParameterVector{arg0, arg1, arg3});
        f->get_ops();
        // Should have thrown, so fail if it didn't
        FAIL() << "Undeclared parameter not detected.";
    } catch (const ov::Exception& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Model references undeclared parameter"));
    } catch (...) {
        FAIL() << "Model construction failed for unexpected reason";
    }
}

// Check no-arg construction
TEST(build_graph, no_arg_construction) {
    // The ops
    // Parameters aren't converted yet
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7});
    auto arg1 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7});
    auto arg3 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{7});
    auto add0 = make_shared<op::v1::Add>();
    auto abs0 = make_shared<op::v0::Abs>();
    auto acos0 = make_shared<op::v0::Acos>();
    auto add1 = make_shared<op::v1::Add>();
    add0->set_argument(1, arg0);
    add0->set_argument(0, arg1);
    abs0->set_argument(0, add0);
    acos0->set_argument(0, add0);
    add1->set_argument(0, acos0);
    add1->set_argument(1, abs0);
    NodeVector ops{arg0, arg1, add0, abs0, acos0, add1};
    for (const auto& op : ov::topological_sort(ops))
        op->revalidate_and_infer_types();
    ASSERT_EQ(add1->get_output_shape(0), Shape{7});
}

TEST(build_graph, multi_output_split_dynamic) {
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    const auto split = make_shared<op::v1::Split>(data, axis, 2);
    auto abs = make_shared<op::v0::Abs>(split->output(1));
    EXPECT_TRUE(abs->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));

    auto new_parameter = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    split->input(0).replace_source_output(new_parameter->output(0));

    auto f = make_shared<Model>(abs, ParameterVector{new_parameter});

    f->validate_nodes_and_infer_types();
    EXPECT_EQ(abs->get_shape(), (Shape{2, 2}));
}

TEST(build_graph, function_revalidate_and_infer) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = op::v0::Constant::create(element::i64, Shape{6}, {1, 3, 16, 2, 2, 2});

    auto r = make_shared<op::v1::Reshape>(arg, pattern, true);
    auto relu = make_shared<op::v0::Relu>(r);
    auto f = make_shared<Model>(relu, ParameterVector{arg});

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));

    auto new_pattern = op::v0::Constant::create(element::i64, Shape{2}, {32, 12});
    r->input(1).replace_source_output(new_pattern->output(0));

    f->validate_nodes_and_infer_types();
    EXPECT_EQ(r->get_output_shape(0), (Shape{32, 12}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{32, 12}));
}

TEST(build_graph, default_output_checks) {
    try {
        std::shared_ptr<Node> empty;
        auto nullout = Output<Node>(empty);
    } catch (...) {
        FAIL() << "nullptr initialization of Output failed";
    }
}

TEST(build_graph, build_graph_with_sink) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<ov::op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<op::v3::Assign>(crop, "v0");

    auto f = make_shared<Model>(ResultVector({res}), SinkVector({assign}), ParameterVector{arg});

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_sink_output_ctor) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<op::v3::Assign>(crop, "v0");

    auto f = make_shared<Model>(OutputVector({pattern->output(0)}), SinkVector({assign}), ParameterVector{arg});

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_add_sink) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<op::v3::Assign>(crop, "v0");

    auto f = make_shared<Model>(ResultVector({res}), ParameterVector{arg});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 5);
    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 0);

    f->add_sinks(SinkVector({assign}));
    sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_wrong_remove_sink) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<ov::op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<op::v3::Assign>(crop, "v0");

    auto f = make_shared<Model>(ResultVector({res}), SinkVector({assign}), ParameterVector{arg});

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    f->remove_sink(assign);
    sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 0);
    auto nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 5);
}

TEST(build_graph, build_graph_with_remove_sink) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<op::v3::Assign>(crop, "v0");

    auto f = make_shared<Model>(ResultVector({res}), SinkVector({assign}), ParameterVector{arg});

    pattern->input(1).replace_source_output(arg);

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    f->remove_sink(assign);
    sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 0);
    auto nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 3);
}

TEST(build_graph, build_graph_with_add_result) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    auto f = make_shared<Model>(ResultVector({res}), ParameterVector{arg});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 5);
    ResultVector results = f->get_results();
    EXPECT_EQ(results.size(), 1);

    f->add_results(ResultVector({res2}));
    results = f->get_results();
    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[1], res2);
    nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_remove_result) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    auto f = make_shared<Model>(ResultVector({res, res2}), ParameterVector{arg});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
    ResultVector results = f->get_results();
    EXPECT_EQ(results.size(), 2);

    f->remove_result(res2);
    results = f->get_results();
    EXPECT_EQ(results.size(), 1);
    nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 5);
}

TEST(build_graph, build_graph_with_add_parameter) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    auto f = make_shared<Model>(ResultVector({res, res2}), ParameterVector{arg});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
    ParameterVector params = f->get_parameters();
    EXPECT_EQ(params.size(), 1);

    pattern->input(1).replace_source_output(arg2->output(0));

    f->add_parameters(ParameterVector({arg2}));
    params = f->get_parameters();
    EXPECT_EQ(params.size(), 2);
    EXPECT_EQ(params[1], arg2);
    nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 7);
}

TEST(build_graph, build_graph_with_remove_parameter) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, arg2};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    auto f = make_shared<Model>(ResultVector({res, res2}), ParameterVector{arg, arg2});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 7);
    ParameterVector params = f->get_parameters();
    EXPECT_EQ(params.size(), 2);

    pattern->input(1).replace_source_output(read->output(0));
    f->remove_parameter(arg2);
    params = f->get_parameters();
    EXPECT_EQ(params.size(), 1);
    nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_remove_parameter_indexing) {
    auto arg = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<op::v3::ReadValue>(init_const, "v0");
    auto assign = make_shared<op::v3::Assign>(read, "v0");
    assign->add_control_dependency(read);
    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    auto f = make_shared<Model>(ResultVector({res, res2}), ParameterVector{arg2, arg});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 7);
    ParameterVector params = f->get_parameters();
    EXPECT_EQ(params.size(), 2);

    pattern->input(0).replace_source_output(read->output(0));
    f->remove_parameter(arg2);
    f->add_sinks(SinkVector{assign});
    params = f->get_parameters();
    EXPECT_EQ(params.size(), 1);
    nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 9);

    f->validate_nodes_and_infer_types();
}

TEST(build_graph, build_graph_parameters_autodetection) {
    // Function with 4 parameters
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, Shape{7, 3});
    auto arg1 = make_shared<op::v0::Parameter>(element::f32, Shape{3});
    auto arg2 = make_shared<op::v0::Parameter>(element::f32, Shape{32, 7});
    auto arg3 = make_shared<op::v0::Parameter>(element::f32, Shape{32, 7});
    auto broadcast_1 = ov::test::utils::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = ov::test::utils::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<op::v0::MatMul>(arg2, arg0);

    auto f = make_shared<Model>(OutputVector{dot});
    EXPECT_EQ(f->get_parameters().size(), 2);
}

TEST(build_graph, build_graph_parameters_variables_autodetection) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});

    auto variable =
        make_shared<op::util::Variable>(op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    auto read = make_shared<op::v6::ReadValue>(init_const, variable);
    auto assign = make_shared<op::v6::Assign>(read, variable);
    assign->add_control_dependency(read);

    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    auto f = make_shared<Model>(OutputVector{res, res2}, SinkVector{assign});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 10);
    ParameterVector params = f->get_parameters();
    EXPECT_EQ(params.size(), 2);
    op::util::VariableVector variables = f->get_variables();
    EXPECT_EQ(variables.size(), 1);
}

TEST(build_graph, build_graph_variables_ctors) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});

    auto variable =
        make_shared<op::util::Variable>(op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    auto read = make_shared<op::v6::ReadValue>(init_const, variable);
    auto assign = make_shared<op::v6::Assign>(read, variable);
    assign->add_control_dependency(read);

    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    {
        auto f = make_shared<Model>(OutputVector{res, res2},
                                    SinkVector{assign},
                                    ParameterVector{arg, arg2},
                                    op::util::VariableVector{variable});

        NodeVector nodes = f->get_ops();
        EXPECT_EQ(nodes.size(), 10);
        ParameterVector params = f->get_parameters();
        EXPECT_EQ(params.size(), 2);
        op::util::VariableVector variables = f->get_variables();
        EXPECT_EQ(variables.size(), 1);
    }

    // autodetect variables
    {
        auto f = make_shared<Model>(OutputVector{res, res2}, SinkVector{assign}, ParameterVector{arg, arg2});
        NodeVector nodes = f->get_ops();
        EXPECT_EQ(nodes.size(), 10);
        ParameterVector params = f->get_parameters();
        EXPECT_EQ(params.size(), 2);
        op::util::VariableVector variables = f->get_variables();
        EXPECT_EQ(variables.size(), 1);
    }
}

TEST(build_graph, build_graph_unregistred_variables) {
    auto arg = make_shared<op::v0::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});

    auto variable =
        make_shared<op::util::Variable>(op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    auto variable_2 =
        make_shared<op::util::Variable>(op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "v1"});
    auto read = make_shared<op::v6::ReadValue>(init_const, variable);
    auto read_2 = make_shared<op::v6::ReadValue>(init_const, variable_2);
    auto assign = make_shared<op::v6::Assign>(read, variable);
    auto assign_2 = make_shared<op::v6::Assign>(read_2, variable_2);
    assign->add_control_dependency(read);

    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<op::v0::Concat>(args, 1);
    auto res = make_shared<op::v0::Result>(pattern);
    const auto axis = op::v0::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::v0::Result>(crop);

    EXPECT_ANY_THROW(const auto unused = make_shared<Model>(OutputVector{res, res2},
                                                            SinkVector{assign, assign_2},
                                                            ParameterVector{arg, arg2},
                                                            op::util::VariableVector{variable}));
}

TEST(build_graph, build_graph_with_sinks_compare) {
    shared_ptr<Model> f0, f1;
    {
        auto init_const0 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto init_const1 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read0 = make_shared<op::v3::ReadValue>(init_const0, "v0");
        auto read1 = make_shared<op::v3::ReadValue>(init_const1, "v1");
        std::vector<shared_ptr<Node>> args = {read0, read1};
        auto add = make_shared<op::v1::Add>(read0, read1);
        auto assign0 = make_shared<op::v3::Assign>(add, "v0");
        auto assign1 = make_shared<op::v3::Assign>(add, "v1");

        f0 = make_shared<Model>(ResultVector({}), SinkVector({assign0, assign1}), ParameterVector{});
    }

    {
        auto init_const0 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto init_const1 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read0 = make_shared<op::v3::ReadValue>(init_const0, "v0");
        auto read1 = make_shared<op::v3::ReadValue>(init_const1, "v1");
        auto add = make_shared<op::v1::Add>(read0, read1);
        auto squeeze = make_shared<op::v0::Squeeze>(add);
        auto assign0 = make_shared<op::v3::Assign>(squeeze, "v0");
        auto assign1 = make_shared<op::v3::Assign>(add, "v1");

        f1 = make_shared<Model>(ResultVector({}), SinkVector({assign0, assign1}), ParameterVector{});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f0, f1);
    EXPECT_FALSE(res.valid) << res.message;
}

TEST(build_graph, build_graph_with_sinks_compare_reads) {
    shared_ptr<Model> f0, f1;
    {
        auto variable0 = make_shared<op::util::Variable>(op::util::VariableInfo{Shape{2, 2}, element::f32, "v0"});
        auto variable1 = make_shared<op::util::Variable>(op::util::VariableInfo{Shape{2, 2}, element::f32, "v1"});

        auto init_const0 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read0 = make_shared<op::v6::ReadValue>(init_const0, variable0);
        auto assign0 = make_shared<op::v6::Assign>(read0, variable0);

        auto init_const1 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read1 = make_shared<op::v6::ReadValue>(init_const1, variable1);
        auto assign1 = make_shared<op::v6::Assign>(read1, variable1);

        f0 = make_shared<Model>(ResultVector({}),
                                SinkVector({assign0, assign1}),
                                ParameterVector{},
                                op::util::VariableVector{variable0, variable1});
    }

    {
        auto variable0 = make_shared<op::util::Variable>(op::util::VariableInfo{Shape{2, 2}, element::f32, "v0"});
        auto variable1 = make_shared<op::util::Variable>(op::util::VariableInfo{Shape{2, 2}, element::f32, "v1"});

        auto init_const0 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read0 = make_shared<op::v6::ReadValue>(init_const0, variable1);
        auto assign0 = make_shared<op::v6::Assign>(read0, variable0);

        auto init_const1 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read1 = make_shared<op::v6::ReadValue>(init_const1, variable0);
        auto assign1 = make_shared<op::v6::Assign>(read1, variable1);

        f1 = make_shared<Model>(ResultVector({}),
                                SinkVector({assign0, assign1}),
                                ParameterVector{},
                                op::util::VariableVector{variable0, variable1});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f0, f1);
    EXPECT_FALSE(res.valid) << res.message;
}

TEST(build_graph, build_graph_with_sinks_compare_results) {
    shared_ptr<Model> f0, f1;
    {
        auto variable0 = make_shared<op::util::Variable>(op::util::VariableInfo{Shape{2, 2}, element::f32, "v0"});
        auto init_const0 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read0 = make_shared<op::v6::ReadValue>(init_const0, variable0);
        auto op = make_shared<op::v0::Relu>(read0);
        auto assign0 = make_shared<op::v6::Assign>(read0, variable0);
        auto result0 = make_shared<op::v0::Result>(assign0);
        auto result1 = make_shared<op::v0::Result>(op);

        f0 = make_shared<Model>(ResultVector({result0, result1}),
                                SinkVector({assign0}),
                                ParameterVector{},
                                op::util::VariableVector{variable0});
    }

    {
        auto variable0 = make_shared<op::util::Variable>(op::util::VariableInfo{Shape{2, 2}, element::f32, "v0"});
        auto init_const0 = op::v0::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
        auto read0 = make_shared<op::v6::ReadValue>(init_const0, variable0);
        auto op = make_shared<op::v0::Relu>(read0);
        auto assign0 = make_shared<op::v6::Assign>(read0, variable0);
        auto result0 = make_shared<op::v0::Result>(assign0);
        auto result1 = make_shared<op::v0::Result>(op);

        f1 = make_shared<Model>(ResultVector({result0, result1}),
                                SinkVector({assign0}),
                                ParameterVector{},
                                op::util::VariableVector{variable0});
    }
    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f0, f1);
    EXPECT_TRUE(res.valid) << res.message;
}
