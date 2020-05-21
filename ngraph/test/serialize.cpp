//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <fstream>
#include <sstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/interpolate.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using json = nlohmann::json;

using ::testing::ElementsAre;
using ::testing::NotNull;
using ::testing::StrEq;

template <typename T>
T get_or_default(nlohmann::json& j, const std::string& key, const T& default_value)
{
    T rc;
    try
    {
        rc = j.at(key).get<T>();
    }
    catch (...)
    {
        rc = default_value;
    }
    return rc;
}

#if defined(NGRAPH_INTERPRETER_ENABLE)
TEST(serialize, main)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, ParameterVector{A, B, C}, "f");

    string js = serialize(f, 4);

    {
        ofstream out("serialize_function.js");
        out << js;
    }

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);
    auto backend = runtime::Backend::create("INTERPRETER");
    auto handle = backend->compile(sfunc);

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    handle->call_with_validate({result}, {x, y, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {y, x, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {x, z, y});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), read_vector<float>(result));
}

TEST(serialize, friendly_name)
{
    // First create "f(A,B,C) = (A+B)*C".
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto sum = A + B;
    auto product = sum * C;
    auto f = make_shared<Function>(product, ParameterVector{A, B, C}, "f");

    A->set_friendly_name("A");
    B->set_friendly_name("B");
    C->set_friendly_name("C");
    sum->set_friendly_name("Sum");
    product->set_friendly_name("Product");

    string js = serialize(f, 4);
    ofstream out("serialize_function.js");
    out << js;

    istringstream in(js);
    shared_ptr<Function> sfunc = deserialize(in);
    auto backend = runtime::Backend::create("INTERPRETER");
    auto handle = backend->compile(sfunc);

    auto x = backend->create_tensor(element::f32, shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->create_tensor(element::f32, shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->create_tensor(element::f32, shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->create_tensor(element::f32, shape);

    handle->call_with_validate({result}, {x, y, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {y, x, z});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), read_vector<float>(result));

    handle->call_with_validate({result}, {x, z, y});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), read_vector<float>(result));
}
#endif

TEST(serialize, existing_models)
{
    vector<string> models = {"mxnet/mnist_mlp_forward.json",
                             "mxnet/10_bucket_LSTM.json",
                             "mxnet/LSTM_backward.json",
                             "mxnet/LSTM_forward.json"};

    for (const string& model : models)
    {
        const string json_path = file_util::path_join(SERIALIZED_ZOO, model);
        const string json_string = file_util::read_file_to_string(json_path);
        shared_ptr<Function> f = ngraph::deserialize(json_string);
    }
}

TEST(serialize, default_value)
{
    json j = {{"test1", 1}, {"test2", 2}};

    int x1 = j.at("test1").get<int>();
    EXPECT_EQ(x1, 1);
    int x2 = get_or_default<int>(j, "test2", 0);
    EXPECT_EQ(x2, 2);
    int x3 = get_or_default<int>(j, "test3", 3);
    EXPECT_EQ(x3, 3);
}

TEST(serialize, constant)
{
    const string tmp_file = "serialize_constant.cpio";
    Shape shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto f = make_shared<Function>(A, ParameterVector{});

    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), A->get_vector<float>());
    serialize(tmp_file, f);
    auto g = deserialize(tmp_file);
    ASSERT_NE(g, nullptr);
    file_util::remove_file(tmp_file);
    bool found = false;
    for (shared_ptr<Node> node : g->get_ops())
    {
        shared_ptr<op::Constant> c = as_type_ptr<op::Constant>(node);
        if (c)
        {
            found = true;
            EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), c->get_vector<float>());
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST(benchmark, serialize)
{
    stopwatch timer;
    string model = "mxnet/LSTM_backward.json";

    const string json_path = file_util::path_join(SERIALIZED_ZOO, model);
    timer.start();
    const string json_string = file_util::read_file_to_string(json_path);
    timer.stop();
    cout << "file read took " << timer.get_milliseconds() << "ms\n";
    timer.start();
    shared_ptr<Function> f = ngraph::deserialize(json_string);
    timer.stop();
    cout << "deserialize took " << timer.get_milliseconds() << "ms\n";

    WithSerializeOutputShapesEnabled serialize_outputs(true);
    ofstream out("test.json");
    out << serialize(f, 4);
}

MATCHER_P2(IsOutputShape, type, shape, "")
{
    return std::get<0>(arg) == type && std::get<1>(arg).to_shape() == shape;
}

TEST(serialize, passthrough)
{
    const string tmp_file = "serialize_passthrough.json";

    using estuple = std::tuple<element::Type, PartialShape>;

    Shape shape{2, 2, 2};
    auto p = make_shared<op::Passthrough>(
        "SerializationTest",
        "Plain",
        "Hello, world!",
        OutputVector{},
        std::vector<estuple>{estuple{element::f32, PartialShape{2, 3}},
                             estuple{element::i8, PartialShape{4, 5}}});
    auto f = make_shared<Function>(NodeVector{std::make_shared<op::GetOutputElement>(p, 0),
                                              std::make_shared<op::GetOutputElement>(p, 1)},
                                   ParameterVector{});
    serialize(tmp_file, f);

    auto g = deserialize(tmp_file);
    file_util::remove_file(tmp_file);
    ASSERT_THAT(g, NotNull());

    std::shared_ptr<op::Passthrough> pt;
    for (const auto& op : g->get_ops())
    {
        pt = as_type_ptr<op::Passthrough>(op);
        if (pt)
        {
            break;
        }
    }
    ASSERT_THAT(pt.get(), NotNull());

    EXPECT_THAT(pt->logical_type(), StrEq("SerializationTest"));
    EXPECT_THAT(pt->language(), StrEq("Plain"));
    EXPECT_THAT(pt->function(), StrEq("Hello, world!"));
    EXPECT_THAT(pt->output_shapes(),
                ElementsAre(IsOutputShape(element::f32, Shape{2, 3}),
                            IsOutputShape(element::i8, Shape{4, 5})));
}

TEST(serialize, constant_infinity_nan)
{
    vector<float> a_data{123.f, 456.f, INFINITY, -INFINITY, NAN};
    vector<float> b_data{5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
    vector<float> c_data{0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05001f, 0.05f};
    vector<int64_t> d_data{-100, -10, -1, 0, 50, 5000000000001};
    auto A = make_shared<op::Constant>(element::f32, Shape{5}, a_data);
    auto B = make_shared<op::Constant>(element::f32, Shape{6}, b_data);
    auto C = make_shared<op::Constant>(element::f32, Shape{7}, c_data);
    auto D = make_shared<op::Constant>(element::i64, Shape{d_data.size()}, d_data);
    A->set_friendly_name("A");
    B->set_friendly_name("B");
    C->set_friendly_name("C");
    D->set_friendly_name("D");
    auto f = make_shared<Function>(NodeVector{A, B, C, D}, ParameterVector{});

    string s = serialize(f, 4);
    shared_ptr<Function> g = deserialize(s);

    shared_ptr<op::Constant> a;
    shared_ptr<op::Constant> b;
    shared_ptr<op::Constant> c;
    shared_ptr<op::Constant> d;
    for (auto node : g->get_ops())
    {
        if (node->get_friendly_name() == "A")
        {
            a = as_type_ptr<op::Constant>(node);
        }
        else if (node->get_friendly_name() == "B")
        {
            b = as_type_ptr<op::Constant>(node);
        }
        else if (node->get_friendly_name() == "C")
        {
            c = as_type_ptr<op::Constant>(node);
        }
        else if (node->get_friendly_name() == "D")
        {
            d = as_type_ptr<op::Constant>(node);
        }
    }
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    ASSERT_TRUE(c);
    ASSERT_TRUE(d);
    EXPECT_TRUE(test::all_close_f(a->get_vector<float>(), a_data));
    EXPECT_TRUE(test::all_close_f(b->get_vector<float>(), b_data));
    EXPECT_TRUE(test::all_close_f(c->get_vector<float>(), c_data));
    EXPECT_EQ(d->get_vector<int64_t>(), d_data);

    string filename = "constant_infinity_nan_test.dot";
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>(filename);
    pass_manager.run_passes(g);
    ifstream file(filename);
    ASSERT_TRUE(file);
    string str((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    EXPECT_NE(str.find(R"(label="A)"), string::npos);
    EXPECT_NE(str.find(R"(label="B)"), string::npos);
    EXPECT_NE(str.find(R"(label="C)"), string::npos);
    EXPECT_NE(str.find(R"(label="D)"), string::npos);
}

TEST(serialize, non_zero_node_output)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{10});
    auto topk = make_shared<op::TopK>(arg, 0, element::i32, 5, true);
    auto abs = make_shared<op::Abs>(Output<Node>(topk, 1));
    auto result = make_shared<op::Result>(abs);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);
    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_abs = g_result->input_value(0).get_node_shared_ptr();
    auto topk_out = g_abs->input_value(0);
    EXPECT_EQ(topk_out.get_index(), 1);
    ASSERT_TRUE(is_type<op::TopK>(topk_out.get_node()));
}

TEST(serialize, opset1_softmax)
{
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{10});
    const auto softmax = make_shared<op::v1::Softmax>(arg, 0);
    const auto result = make_shared<op::Result>(softmax);
    const auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    const auto g_result = g->get_results().at(0);
    const auto g_softmax = g_result->get_input_node_shared_ptr(0);
    EXPECT_TRUE(is_type<op::v1::Softmax>(g_softmax));
}

TEST(serialize, opset1_gather)
{
    auto params = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto axis = make_shared<op::Parameter>(element::i64, Shape{1});
    auto gather_v1 = make_shared<op::v1::Gather>(params, indices, axis);

    auto result = make_shared<op::Result>(gather_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{params, indices, axis});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_gather = g_result->get_input_node_shared_ptr(0);
    EXPECT_TRUE(is_type<op::v1::Gather>(g_gather));
}

TEST(serialize, opset1_product)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto keep_dims = true;
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto reduce_prod = make_shared<op::v1::ReduceProd>(arg, axes, keep_dims);
    auto result = make_shared<op::Result>(reduce_prod);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_red_prod = g_result->get_input_node_shared_ptr(0);
    auto node = as_type_ptr<op::v1::ReduceProd>(g_red_prod);
    EXPECT_TRUE(node);
    EXPECT_EQ(node->get_keep_dims(), 1);
    EXPECT_EQ(node->get_reduction_axes(), AxisSet({1, 2}));
}

TEST(serialize, opset1_sum)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto keep_dims = true;
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    auto reduce_sum = make_shared<op::v1::ReduceSum>(arg, axes, keep_dims);
    auto result = make_shared<op::Result>(reduce_sum);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_red_sum = g_result->get_input_node_shared_ptr(0);
    auto node = as_type_ptr<op::v1::ReduceSum>(g_red_sum);
    EXPECT_TRUE(node);
    EXPECT_EQ(node->get_keep_dims(), 1);
    EXPECT_EQ(node->get_reduction_axes(), AxisSet({1, 2}));
}

TEST(serialize, opset1_pad)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{4, 5, 6});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{2});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});
    auto pad_mode = op::PadMode::EDGE;
    auto pad = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, pad_mode);

    auto result = make_shared<op::Result>(pad);
    auto f = make_shared<Function>(ResultVector{result},
                                   ParameterVector{arg, pads_begin, pads_end, arg_pad_value});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_pad = as_type_ptr<op::v1::Pad>(g_result->input_value(0).get_node_shared_ptr());
    ASSERT_TRUE(g_pad);
    EXPECT_EQ(g_pad->get_pad_mode(), pad_mode);
}

TEST(serialize, tensor_iterator_raw)
{
    // That which we iterate over
    auto X = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});

    // Common to all cells
    auto WH = make_shared<op::Parameter>(element::f32, Shape{20, 20});
    auto WX = make_shared<op::Parameter>(element::f32, Shape{10, 20});
    auto bH = make_shared<op::Parameter>(element::f32, Shape{20});
    auto WY = make_shared<op::Parameter>(element::f32, Shape{20, 5});
    auto bY = make_shared<op::Parameter>(element::f32, Shape{5});

    // Initial values
    auto Hinit = make_shared<op::Parameter>(element::f32, Shape{32, 1, 20});

    // Set up the cell body, a function from (Hi, Xi) -> (Ho, Yo)
    // Cell parameters
    auto Hi = make_shared<op::Parameter>(element::f32, Shape{32, 1, 20});
    auto Xi = make_shared<op::Parameter>(element::f32, Shape{32, 1, 10});
    auto WH_body = make_shared<op::Parameter>(element::f32, Shape{20, 20});
    auto WX_body = make_shared<op::Parameter>(element::f32, Shape{10, 20});
    auto bH_body = make_shared<op::Parameter>(element::f32, Shape{20});
    auto WY_body = make_shared<op::Parameter>(element::f32, Shape{20, 5});
    auto bY_body = make_shared<op::Parameter>(element::f32, Shape{5});

    // Body
    auto Ho = make_shared<op::Reshape>(
        make_shared<op::Relu>(
            make_shared<op::Dot>(make_shared<op::Reshape>(Xi, AxisVector{0, 1, 2}, Shape{32, 10}),
                                 WX_body) +
            make_shared<op::Dot>(make_shared<op::Reshape>(Hi, AxisVector{0, 1, 2}, Shape{32, 20}),
                                 WH_body) +
            make_shared<op::Broadcast>(bH_body, Shape{32, 20}, AxisSet{0})),
        AxisVector{0, 1},
        Shape{32, 1, 20});
    auto Yo = make_shared<op::Relu>(
        make_shared<op::Dot>(make_shared<op::Reshape>(Ho, AxisVector{0, 1, 2}, Shape{32, 20}),
                             WY_body) +
        make_shared<op::Broadcast>(bY_body, Shape{32, 5}, AxisSet{0}));
    auto body = make_shared<op::TensorIterator::BodyLambda>(
        OutputVector{Yo, Ho}, ParameterVector{Xi, Hi, WH_body, WX_body, WY_body, bH_body, bY_body});

    auto tensor_iterator = make_shared<op::TensorIterator>();
    tensor_iterator->set_body(body);
    // The Xi are the elements of Xseq
    // start=0, stride=1, part_size=1, end=39, axis=1
    tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, 39, 1);
    // Hi is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(Hi, Hinit, Ho);
    tensor_iterator->set_invariant_input(WH_body, WH);
    tensor_iterator->set_invariant_input(WX_body, WX);
    tensor_iterator->set_invariant_input(WY_body, WY);
    tensor_iterator->set_invariant_input(bH_body, bH);
    tensor_iterator->set_invariant_input(bY_body, bY);

    // Output 0 is last Yo
    auto out0 = tensor_iterator->get_iter_value(Yo, -1);
    // Output 1 is concat of hidden states
    // start=0, stride=1, part_size=1, end=39, axis=1
    auto out1 = tensor_iterator->get_concatenated_slices(Ho, 0, 1, 1, 39, 1);

    auto results = ResultVector{make_shared<op::Result>(out0), make_shared<op::Result>(out1)};
    auto f = make_shared<Function>(results, ParameterVector{X, Hinit, WH, WX, bH, WY, bY});
    string s = serialize(f);
    shared_ptr<Function> g = deserialize(s);

    ngraph::test::NodeBuilder builder;
    // Uncomment to see serialization
    // builder.set_print(true);
    builder.save_node(tensor_iterator);
    auto g_tensor_iterator = as_type_ptr<op::v0::TensorIterator>(builder.create());
    ASSERT_TRUE(g_tensor_iterator);
    auto& inputs = tensor_iterator->get_input_descriptions();
    auto& g_inputs = g_tensor_iterator->get_input_descriptions();
    ASSERT_EQ(inputs.size(), g_inputs.size());
    for (size_t i = 0; i < tensor_iterator->get_input_descriptions().size(); ++i)
    {
        auto& val = inputs[i];
        auto& g_val = g_inputs[i];
        ASSERT_EQ(val->get_type_info(), g_val->get_type_info());
        ASSERT_EQ(val->m_input_index, g_val->m_input_index);
        ASSERT_EQ(val->m_body_parameter_index, g_val->m_body_parameter_index);
    }
    auto& outputs = tensor_iterator->get_output_descriptions();
    auto& g_outputs = g_tensor_iterator->get_output_descriptions();
    ASSERT_EQ(outputs.size(), g_outputs.size());
    for (size_t i = 0; i < tensor_iterator->get_output_descriptions().size(); ++i)
    {
        auto& val = outputs[i];
        auto& g_val = g_outputs[i];
        ASSERT_EQ(val->get_type_info(), g_val->get_type_info());
    }
}

TEST(serialize, tensor_iterator_lstm)
{
    // That which we iterate over
    const size_t N = 32; // Batch size
    const size_t L = 10; // Sequence length
    const size_t I = 8;  // Input size
    const size_t H = 32; // Hidden size
    auto SENT = make_shared<op::Parameter>(element::f32, Shape{N, L, I});

    auto H_init = make_shared<op::Parameter>(element::f32, Shape{N, 1, H});
    auto C_init = make_shared<op::Parameter>(element::f32, Shape{N, 1, H});

    auto W = make_shared<op::Parameter>(element::f32, Shape{4 * H, I});
    auto R = make_shared<op::Parameter>(element::f32, Shape{4 * H, H});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{N, 1, H});
    auto C_t = make_shared<op::Parameter>(element::f32, Shape{N, 1, H});

    // Body
    auto X = make_shared<op::Parameter>(element::f32, Shape{N, 1, I});
    auto W_body = make_shared<op::Parameter>(element::f32, Shape{4 * H, I});
    auto R_body = make_shared<op::Parameter>(element::f32, Shape{4 * H, H});
    auto LSTM_cell =
        make_shared<op::LSTMCell>(make_shared<op::Reshape>(X, AxisVector{0, 1, 2}, Shape{N, I}),
                                  make_shared<op::Reshape>(H_t, AxisVector{0, 1, 2}, Shape{N, H}),
                                  make_shared<op::Reshape>(C_t, AxisVector{0, 1, 2}, Shape{N, H}),
                                  W_body,
                                  R_body,
                                  H);
    auto H_o = make_shared<op::Reshape>(LSTM_cell->output(0), AxisVector{0, 1}, Shape{N, 1, H});
    auto C_o = make_shared<op::Reshape>(LSTM_cell->output(1), AxisVector{0, 1}, Shape{N, 1, H});
    auto body = make_shared<op::TensorIterator::BodyLambda>(
        OutputVector{H_o, C_o}, ParameterVector{X, H_t, C_t, W_body, R_body});

    auto tensor_iterator = make_shared<op::TensorIterator>();
    tensor_iterator->set_body(body);
    // start=0, stride=1, part_size=1, end=39, axis=1
    tensor_iterator->set_sliced_input(X, SENT, 0, 1, 1, -1, 1);
    // H_t is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);
    tensor_iterator->set_invariant_input(W_body, W);
    tensor_iterator->set_invariant_input(R_body, R);

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results = ResultVector{make_shared<op::Result>(out0), make_shared<op::Result>(out1)};
    auto f = make_shared<Function>(results, ParameterVector{SENT, H_init, C_init, W, R});
    string s = serialize(f);
    shared_ptr<Function> g = deserialize(s);
}

TEST(serialize, tensor_iterator_2_slice_inputs_part_size_2)
{
    // That which we iterate over
    auto X = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});
    auto M = make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});
    auto Yi = make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});
    auto M_body = make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});

    // Body
    auto Zo = (Xi + Yi) * M_body;
    auto body = make_shared<op::TensorIterator::BodyLambda>(OutputVector{Zo},
                                                            ParameterVector{Xi, Yi, M_body});

    auto tensor_iterator = make_shared<op::TensorIterator>();
    tensor_iterator->set_body(body);
    // The Xi are the elements of Xseq
    // start=0, stride=2, part_size=2, end=39, axis=1
    tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 39, 1);
    // The Yi are the elements of Yseq
    // start=0, stride=2, part_size=2, end=-1, axis=1
    tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -1, 1);
    tensor_iterator->set_invariant_input(M_body, M);

    // Output 0 is last Zo
    auto out0 = tensor_iterator->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=2, part_size=2, end=39, axis=1
    auto out1 = tensor_iterator->get_concatenated_slices(Zo, 0, 2, 2, 39, 1);

    auto result0 = make_shared<op::Result>(out0);
    auto result1 = make_shared<op::Result>(out1);
    Shape out0_shape{32, 2, 10};
    Shape out1_shape{32, 40, 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);

    string s = serialize(f);
    shared_ptr<Function> g = deserialize(s);
}

TEST(serialize, tensor_iterator_2_slice_inputs_part_size_2_dynamic)
{
    // That which we iterate over
    auto X = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});
    auto M = make_shared<op::Parameter>(element::f32, Shape{32, 2, 10});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xi = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Yi = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto M_body = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());

    // Body
    auto Zo = (Xi + Yi) * M_body;
    auto body = make_shared<op::TensorIterator::BodyLambda>(OutputVector{Zo},
                                                            ParameterVector{Xi, Yi, M_body});

    auto tensor_iterator = make_shared<op::TensorIterator>();
    tensor_iterator->set_body(body);
    // The Xi are the elements of Xseq
    // start=0, stride=2, part_size=2, end=38, axis=1
    tensor_iterator->set_sliced_input(Xi, X, 0, 2, 2, 38, 1);
    // The Yi are the elements of Yseq
    // start=0, stride=2, part_size=2, end=-2, axis=1
    tensor_iterator->set_sliced_input(Yi, Y, 0, 2, 2, -2, 1);
    tensor_iterator->set_invariant_input(M_body, M);

    // check input descriptors
    for (auto& desc : tensor_iterator->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::op::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc = as_type_ptr<ngraph::op::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc = as_type_ptr<ngraph::op::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = tensor_iterator->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=2, part_size=2, end=38, axis=1
    auto out1 = tensor_iterator->get_concatenated_slices(Zo, 0, 2, 2, 38, 1);

    // check output descriptors
    for (auto& desc : tensor_iterator->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::op::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc = as_type_ptr<ngraph::op::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }

    auto result0 = make_shared<op::Result>(out0);
    auto result1 = make_shared<op::Result>(out1);
    Shape out0_shape{32, 2, 10};
    Shape out1_shape{32, 38, 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);

    EXPECT_EQ(body->get_results()[0]->get_output_shape(0), out0_shape);

    string s = serialize(f);
    shared_ptr<Function> g = deserialize(s);
}

TEST(serialize, opset1_strided_slice)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4, 6, 8});
    auto begin = make_shared<op::Parameter>(element::i64, Shape{4});
    auto end = make_shared<op::Parameter>(element::i64, Shape{4});
    auto strides = make_shared<op::Parameter>(element::i64, Shape{4});

    const std::vector<int64_t> begin_mask{1, 0, 1, 0};
    const std::vector<int64_t> end_mask{1, 1, 1, 0};
    const std::vector<int64_t> new_axis_mask{0, 0, 1, 1};
    const std::vector<int64_t> shrink_axis_mask{0, 0, 0, 0};
    const std::vector<int64_t> ellipsis_mask{1, 1, 1, 1};

    auto strided_slice_in = make_shared<op::v1::StridedSlice>(data,
                                                              begin,
                                                              end,
                                                              strides,
                                                              begin_mask,
                                                              end_mask,
                                                              new_axis_mask,
                                                              shrink_axis_mask,
                                                              ellipsis_mask);

    auto result = make_shared<op::Result>(strided_slice_in);
    auto f =
        make_shared<Function>(ResultVector{result}, ParameterVector{data, begin, end, strides});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_strided_slice_v1 = g_result->get_input_node_shared_ptr(0);
    auto strided_slice_out = as_type_ptr<op::v1::StridedSlice>(g_strided_slice_v1);

    ASSERT_TRUE(strided_slice_out);
    EXPECT_EQ(strided_slice_out->get_begin_mask(), begin_mask);
    EXPECT_EQ(strided_slice_out->get_end_mask(), end_mask);
    EXPECT_EQ(strided_slice_out->get_new_axis_mask(), new_axis_mask);
    EXPECT_EQ(strided_slice_out->get_shrink_axis_mask(), shrink_axis_mask);
    EXPECT_EQ(strided_slice_out->get_ellipsis_mask(), ellipsis_mask);
}

TEST(serialize, opset1_binary_convolution)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 2});
    auto filter = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 2});
    const Strides strides{1, 1};
    const CoordinateDiff pads_begin{0, 0};
    const CoordinateDiff pads_end{0, 0};
    const Strides dilations{1, 1};
    auto mode = op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT;
    const float pad_value = 2.1f;
    const auto auto_pad = op::PadType::NOTSET;

    auto binary_conv_in = make_shared<op::v1::BinaryConvolution>(
        data, filter, strides, pads_begin, pads_end, dilations, mode, pad_value, auto_pad);

    auto result = make_shared<op::Result>(binary_conv_in);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, filter});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_binary_conv = g_result->get_input_node_shared_ptr(0);
    auto binary_conv_out = as_type_ptr<op::v1::BinaryConvolution>(g_binary_conv);
    ASSERT_TRUE(binary_conv_out);

    EXPECT_EQ(binary_conv_out->get_strides(), strides);
    EXPECT_EQ(binary_conv_out->get_pads_begin(), pads_begin);
    EXPECT_EQ(binary_conv_out->get_pads_end(), pads_end);
    EXPECT_EQ(binary_conv_out->get_dilations(), dilations);
    EXPECT_EQ(binary_conv_out->get_mode(),
              op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT);
    EXPECT_EQ(binary_conv_out->get_pad_value(), pad_value);
    EXPECT_EQ(binary_conv_out->get_auto_pad(), auto_pad);
}

TEST(serialize, opset1_interpolate)
{
    auto image = make_shared<op::Parameter>(element::f32, Shape{2, 2, 33, 65});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {15, 30});
    op::InterpolateAttrs attrs;
    attrs.axes = {2, 3};
    attrs.mode = "linear";
    attrs.align_corners = true;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto op = make_shared<op::Interpolate>(image, output_shape, attrs);
    auto result = make_shared<op::Result>(op);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{image});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_interpolate = g_result->get_input_node_shared_ptr(0);
    auto g_op = as_type_ptr<op::Interpolate>(g_interpolate);
    ASSERT_TRUE(g_op);
    op::InterpolateAttrs g_attrs = g_op->get_attrs();
    EXPECT_EQ(g_attrs.axes, attrs.axes);
    EXPECT_EQ(g_attrs.mode, attrs.mode);
    EXPECT_EQ(g_attrs.align_corners, attrs.align_corners);
    EXPECT_EQ(g_attrs.antialias, attrs.antialias);
    EXPECT_EQ(g_attrs.pads_begin, attrs.pads_begin);
    EXPECT_EQ(g_attrs.pads_end, attrs.pads_end);
}

TEST(serialize, opset3_interpolate)
{
    using op::v3::Interpolate;
    using InterpolateMode = op::v3::Interpolate::InterpolateMode;
    using CoordinateTransformMode = op::v3::Interpolate::CoordinateTransformMode;
    using InterpolateAttrs = op::v3::Interpolate::InterpolateAttrs;

    auto image = make_shared<op::Parameter>(element::f32, Shape{2, 2, 33, 65});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {15, 30});
    InterpolateAttrs attrs;
    attrs.axes = {2, 3};
    attrs.mode = InterpolateMode::linear;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::half_pixel;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto op = make_shared<Interpolate>(image, output_shape, attrs);
    auto result = make_shared<op::Result>(op);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{image});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_interpolate = g_result->get_input_node_shared_ptr(0);
    auto g_op = as_type_ptr<op::v3::Interpolate>(g_interpolate);
    ASSERT_TRUE(g_op);
    InterpolateAttrs g_attrs = g_op->get_attrs();
    EXPECT_EQ(g_attrs.axes, attrs.axes);
    EXPECT_EQ(g_attrs.mode, attrs.mode);
    EXPECT_EQ(g_attrs.coordinate_transformation_mode, attrs.coordinate_transformation_mode);
    EXPECT_EQ(g_attrs.antialias, attrs.antialias);
    EXPECT_EQ(g_attrs.pads_begin, attrs.pads_begin);
    EXPECT_EQ(g_attrs.pads_end, attrs.pads_end);
}

TEST(serialize, depth_to_space)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{4, 5, 6});
    auto mode = op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    size_t block_size = 2;
    auto depth_to_space_in = make_shared<op::DepthToSpace>(arg, mode, block_size);

    auto result = make_shared<op::Result>(depth_to_space_in);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_depth_to_space = g_result->get_input_node_shared_ptr(0);
    auto depth_to_space_out = as_type_ptr<op::DepthToSpace>(g_depth_to_space);
    ASSERT_TRUE(depth_to_space_out);
    EXPECT_EQ(depth_to_space_out->get_block_size(), block_size);
    EXPECT_EQ(depth_to_space_out->get_mode(), mode);
}

TEST(serialize, space_to_depth)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{4, 6, 8});
    auto mode = op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    size_t block_size = 2;
    auto space_to_depth_in = make_shared<op::SpaceToDepth>(arg, mode, block_size);

    auto result = make_shared<op::Result>(space_to_depth_in);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_space_to_depth = g_result->get_input_node_shared_ptr(0);
    auto depth_to_space_out = as_type_ptr<op::SpaceToDepth>(g_space_to_depth);
    ASSERT_TRUE(depth_to_space_out);
    EXPECT_EQ(depth_to_space_out->get_block_size(), block_size);
    EXPECT_EQ(depth_to_space_out->get_mode(), mode);
}

TEST(serialize, deformable_psroi_pooling)
{
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{1, 1});
    auto offsets = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const int64_t output_dim = 1;
    const int64_t group_size = 2;
    const float spatial_scale = 3;
    std::string mode = "bilinear_deformable";
    int64_t spatial_bins_x = 4;
    int64_t spatial_bins_y = 5;
    float trans_std = 6.1f;
    int64_t part_size = 7;

    auto def_psroi_pool_in = make_shared<op::v1::DeformablePSROIPooling>(input,
                                                                         coords,
                                                                         offsets,
                                                                         output_dim,
                                                                         spatial_scale,
                                                                         group_size,
                                                                         mode,
                                                                         spatial_bins_x,
                                                                         spatial_bins_y,
                                                                         trans_std,
                                                                         part_size);

    auto result = make_shared<op::Result>(def_psroi_pool_in);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{input, coords, offsets});
    string s = serialize(f);

    shared_ptr<Function> g = deserialize(s);
    auto g_result = g->get_results().at(0);
    auto g_def_psroi_pool = g_result->get_input_node_shared_ptr(0);
    auto def_psroi_pool_out = as_type_ptr<op::v1::DeformablePSROIPooling>(g_def_psroi_pool);

    EXPECT_EQ(def_psroi_pool_out->description(), "DeformablePSROIPooling");
    EXPECT_EQ(def_psroi_pool_out->get_version(), 1);

    EXPECT_EQ(def_psroi_pool_out->get_output_dim(), output_dim);
    EXPECT_EQ(def_psroi_pool_out->get_group_size(), group_size);
    EXPECT_EQ(def_psroi_pool_out->get_spatial_scale(), spatial_scale);
    EXPECT_EQ(def_psroi_pool_out->get_mode(), mode);
    EXPECT_EQ(def_psroi_pool_out->get_spatial_bins_x(), spatial_bins_x);
    EXPECT_EQ(def_psroi_pool_out->get_spatial_bins_y(), spatial_bins_y);
    EXPECT_EQ(def_psroi_pool_out->get_trans_std(), trans_std);
    EXPECT_EQ(def_psroi_pool_out->get_part_size(), part_size);
}
