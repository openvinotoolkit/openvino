// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "util/test_tools.hpp"

#include <memory>
#include <util/type_prop.hpp>

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(build_graph, build_simple)
{
    // Function with 4 parameters
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{7, 3});
    auto arg1 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{32, 7});
    auto arg3 = make_shared<op::Parameter>(element::f32, Shape{32, 7});
    auto broadcast_1 = builder::opset1::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = builder::opset1::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<op::MatMul>(arg2, arg0);
    ASSERT_EQ(dot->input_value(0).get_node_shared_ptr(), arg2);
    ASSERT_EQ(dot->input_value(1).get_node_shared_ptr(), arg0);

    auto cluster_0 = make_shared<Function>(dot, ParameterVector{arg0, arg1, arg2, arg3});

    ASSERT_EQ(cluster_0->get_output_op(0)->input_value(0).get_node_shared_ptr(), dot);
}

TEST(build_graph, literal)
{
    // float scalar from a float
    // auto float0 = FloatConstant::make(3.0);
    vector<float> float_t{3.0};
    auto float0 = make_shared<op::Constant>(element::f32, Shape{1}, float_t);
    ASSERT_EQ(float0->get_vector<float>(), std::vector<float>{3.0});
    ASSERT_EQ(float0->get_element_type(), element::f32);
    ASSERT_EQ(float0->get_shape(), Shape{1});
    auto d = make_shared<op::MatMul>(float0, float0);
    ASSERT_EQ(d->input_values().at(0).get_node_shared_ptr(), float0);
    ASSERT_EQ(d->input_values().at(1).get_node_shared_ptr(), float0);

    vector<int32_t> int32{3};
    auto int32_0 = make_shared<op::Constant>(element::i32, Shape{}, int32);
    ASSERT_EQ(int32_0->get_vector<int32_t>(), std::vector<int>{3});
    ASSERT_EQ(int32_0->get_element_type(), element::i32);
    ASSERT_EQ(int32_0->get_shape(), Shape{});
}

TEST(build_graph, tensor)
{
    // float scalar from a float
    // auto float0 = FloatConstant::make(3.0);
    Shape shape{2, 3};
    vector<float> float_t(shape_size(shape), 0);
    auto float0 = make_shared<op::Constant>(element::f32, shape, float_t);
    ASSERT_EQ(float0->get_element_type(), element::f32);
    ASSERT_EQ(float0->get_shape(), shape);
    auto d = make_shared<op::v1::Add>(float0, float0);
    ASSERT_EQ(d->input_values().at(0).get_node_shared_ptr(), float0);
    ASSERT_EQ(d->input_values().at(1).get_node_shared_ptr(), float0);

    Shape ishape{3, 5};
    vector<int32_t> idata(shape_size(ishape), 0);
    auto int32_0 = make_shared<op::Constant>(element::i32, ishape, idata);
    ASSERT_EQ(int32_0->get_element_type(), element::i32);
    ASSERT_EQ(int32_0->get_shape(), ishape);
}

// Check functions with undeclared parameters
TEST(build_graph, function_undeclared_parameters)
{
    // Function with 4 parameters
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{7, 3});
    auto arg1 = make_shared<op::Parameter>(element::f32, Shape{3});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{32, 7});
    auto arg3 = make_shared<op::Parameter>(element::f32, Shape{32, 7});
    auto broadcast_1 = builder::opset1::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = builder::opset1::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<op::MatMul>(arg2, arg0);
    ASSERT_EQ(dot->input_values()[0].get_node_shared_ptr(), arg2);
    ASSERT_EQ(dot->input_values()[1].get_node_shared_ptr(), arg0);
    try
    {
        auto f = make_shared<Function>(dot, ParameterVector{arg0, arg1, arg3});
        f->get_ops();
        // Should have thrown, so fail if it didn't
        FAIL() << "Undeclared parameter not detected.";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Function references undeclared parameter"));
    }
    catch (...)
    {
        FAIL() << "Function construction failed for unexpected reason";
    }
}

// Check no-arg construction
TEST(build_graph, no_arg_construction)
{
    // The ops
    // Parameters aren't converted yet
    auto arg0 = make_shared<op::Parameter>(element::f32, Shape{7});
    auto arg1 = make_shared<op::Parameter>(element::f32, Shape{7});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{7});
    auto arg3 = make_shared<op::Parameter>(element::f32, Shape{7});
    auto add0 = make_shared<op::v1::Add>();
    auto abs0 = make_shared<op::Abs>();
    auto acos0 = make_shared<op::Acos>();
    auto add1 = make_shared<op::v1::Add>();
    add0->set_argument(1, arg0);
    add0->set_argument(0, arg1);
    abs0->set_argument(0, add0);
    acos0->set_argument(0, add0);
    add1->set_argument(0, acos0);
    add1->set_argument(1, abs0);
    NodeVector ops{arg0, arg1, add0, abs0, acos0, add1};
    validate_nodes_and_infer_types(ops);
    ASSERT_EQ(add1->get_output_shape(0), Shape{7});
}

TEST(build_graph, multi_output_split_dynamic)
{
    const auto data = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    const auto split = make_shared<op::v1::Split>(data, axis, 2);
    auto abs = make_shared<op::Abs>(split->output(1));
    EXPECT_TRUE(abs->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));

    auto new_parameter = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    split->input(0).replace_source_output(new_parameter->output(0));

    auto f = make_shared<Function>(abs, ParameterVector{new_parameter});

    f->validate_nodes_and_infer_types();
    EXPECT_EQ(abs->get_shape(), (Shape{2, 2}));
}

TEST(build_graph, function_revalidate_and_infer)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto pattern = op::Constant::create(element::i64, Shape{6}, {1, 3, 16, 2, 2, 2});

    auto r = make_shared<op::v1::Reshape>(arg, pattern, true);
    auto relu = make_shared<op::Relu>(r);
    auto f = make_shared<Function>(relu, ParameterVector{arg});

    EXPECT_EQ(r->get_output_element_type(0), element::f32);
    EXPECT_EQ(r->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{1, 3, 16, 2, 2, 2}));

    auto new_pattern = op::Constant::create(element::i64, Shape{2}, {32, 12});
    r->input(1).replace_source_output(new_pattern->output(0));

    f->validate_nodes_and_infer_types();
    EXPECT_EQ(r->get_output_shape(0), (Shape{32, 12}));
    EXPECT_EQ(f->get_output_shape(0), (Shape{32, 12}));
}

TEST(build_graph, default_output_checks)
{
    try
    {
        std::shared_ptr<Node> empty;
        auto nullout = Output<Node>(empty);
    }
    catch (...)
    {
        FAIL() << "nullptr initialization of Output failed";
    }
}

TEST(build_graph, build_graph_with_sink)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<opset5::Assign>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res}), SinkVector({assign}), ParameterVector{arg});

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_sink_output_ctor)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<opset5::Assign>(crop, "v0");

    auto f = make_shared<Function>(
        OutputVector({pattern->output(0)}), SinkVector({assign}), ParameterVector{arg});

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 8);
}

TEST(build_graph, build_graph_with_add_sink)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<opset5::Assign>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res}), ParameterVector{arg});

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

TEST(build_graph, build_graph_with_wrong_remove_sink)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<opset5::Assign>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res}), SinkVector({assign}), ParameterVector{arg});

    SinkVector sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 1);
    EXPECT_EQ(sinks[0], assign);
    f->remove_sink(assign);
    sinks = f->get_sinks();
    EXPECT_EQ(sinks.size(), 0);
    auto nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 5);
}

TEST(build_graph, build_graph_with_remove_sink)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto assign = make_shared<opset5::Assign>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res}), SinkVector({assign}), ParameterVector{arg});

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

TEST(build_graph, build_graph_with_add_result)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::Result>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res}), ParameterVector{arg});

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

TEST(build_graph, build_graph_with_remove_result)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::Result>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res, res2}), ParameterVector{arg});

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

TEST(build_graph, build_graph_with_add_parameter)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, read};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::Result>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res, res2}), ParameterVector{arg});

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

TEST(build_graph, build_graph_with_remove_parameter)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    std::vector<shared_ptr<Node>> args = {arg, arg2};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::Result>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res, res2}), ParameterVector{arg, arg2});

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

TEST(build_graph, build_graph_with_remove_parameter_indexing)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto init_const = op::Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});
    auto read = make_shared<opset5::ReadValue>(init_const, "v0");
    auto assign = make_shared<opset5::Assign>(read, "v0");
    assign->add_control_dependency(read);
    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<op::Concat>(args, 1);
    auto res = make_shared<op::Result>(pattern);
    const auto axis = op::Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<op::v1::Split>(pattern, axis, 3);
    auto res2 = make_shared<op::Result>(crop, "v0");

    auto f = make_shared<Function>(ResultVector({res, res2}), ParameterVector{arg2, arg});

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

TEST(build_graph, build_graph_parameters_autodetection)
{
    // Function with 4 parameters
    using namespace opset7;
    auto arg0 = make_shared<Parameter>(element::f32, Shape{7, 3});
    auto arg1 = make_shared<Parameter>(element::f32, Shape{3});
    auto arg2 = make_shared<Parameter>(element::f32, Shape{32, 7});
    auto arg3 = make_shared<Parameter>(element::f32, Shape{32, 7});
    auto broadcast_1 = builder::opset1::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto b1 = builder::opset1::make_broadcast(arg3, Shape{10, 32, 7}, AxisSet{0});
    auto dot = make_shared<MatMul>(arg2, arg0);

    auto f = make_shared<Function>(OutputVector{dot});
    EXPECT_EQ(f->get_parameters().size(), 2);
}

TEST(build_graph, build_graph_parameters_variables_autodetection)
{
    using namespace opset7;
    auto arg = make_shared<Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<Parameter>(element::f32, Shape{2, 2});
    auto init_const = Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});

    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    auto read = make_shared<ReadValue>(init_const, variable);
    auto assign = make_shared<Assign>(read, variable);
    assign->add_control_dependency(read);

    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<Concat>(args, 1);
    auto res = make_shared<Result>(pattern);
    const auto axis = Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<Split>(pattern, axis, 3);
    auto res2 = make_shared<Result>(crop, "v0");

    auto f = make_shared<Function>(OutputVector{res, res2}, SinkVector{assign});

    NodeVector nodes = f->get_ops();
    EXPECT_EQ(nodes.size(), 10);
    ParameterVector params = f->get_parameters();
    EXPECT_EQ(params.size(), 2);
    VariableVector variables = f->get_variables();
    EXPECT_EQ(variables.size(), 1);
}

TEST(build_graph, build_graph_variables_ctors)
{
    using namespace opset7;
    auto arg = make_shared<Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<Parameter>(element::f32, Shape{2, 2});
    auto init_const = Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});

    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    auto read = make_shared<ReadValue>(init_const, variable);
    auto assign = make_shared<Assign>(read, variable);
    assign->add_control_dependency(read);

    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<Concat>(args, 1);
    auto res = make_shared<Result>(pattern);
    const auto axis = Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<Split>(pattern, axis, 3);
    auto res2 = make_shared<Result>(crop, "v0");

    {
        auto f = make_shared<Function>(OutputVector{res, res2}, SinkVector{assign},
                                       ParameterVector{arg, arg2}, VariableVector{variable});

        NodeVector nodes = f->get_ops();
        EXPECT_EQ(nodes.size(), 10);
        ParameterVector params = f->get_parameters();
        EXPECT_EQ(params.size(), 2);
        VariableVector variables = f->get_variables();
        EXPECT_EQ(variables.size(), 1);
    }

    // autodetect variables
    {
        auto f = make_shared<Function>(OutputVector{res, res2}, SinkVector{assign},
                                         ParameterVector{arg, arg2});
        NodeVector nodes = f->get_ops();
        EXPECT_EQ(nodes.size(), 10);
        ParameterVector params = f->get_parameters();
        EXPECT_EQ(params.size(), 2);
        VariableVector variables = f->get_variables();
        EXPECT_EQ(variables.size(), 1);
    }
}

TEST(build_graph, build_graph_unregistred_variables)
{
    using namespace opset7;
    auto arg = make_shared<Parameter>(element::f32, Shape{2, 4});
    auto arg2 = make_shared<Parameter>(element::f32, Shape{2, 2});
    auto init_const = Constant::create(element::f32, Shape{2, 2}, {0, 0, 0, 0});

    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v0"});
    auto variable_2 = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "v1"});
    auto read = make_shared<ReadValue>(init_const, variable);
    auto read_2 = make_shared<ReadValue>(init_const, variable_2);
    auto assign = make_shared<Assign>(read, variable);
    auto assign_2 = make_shared<Assign>(read_2, variable_2);
    assign->add_control_dependency(read);

    std::vector<shared_ptr<Node>> args = {arg2, arg};
    auto pattern = make_shared<Concat>(args, 1);
    auto res = make_shared<Result>(pattern);
    const auto axis = Constant::create(element::i64, Shape{}, {1});
    auto crop = make_shared<Split>(pattern, axis, 3);
    auto res2 = make_shared<Result>(crop, "v0");

    EXPECT_ANY_THROW(make_shared<Function>(OutputVector{res, res2}, SinkVector{assign, assign_2},
                                   ParameterVector{arg, arg2}, VariableVector{variable}));
}