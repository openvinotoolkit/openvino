//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/op/add.hpp"
#include "ngraph/op/assign.hpp"
#include "ngraph/op/read_value.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, partial_shapes)
{
    auto pshape = PartialShape::dynamic();
    ASSERT_TRUE(pshape.compatible(PartialShape{1, 32}));
}

TEST(op_eval, read_value_and_assign)
{
    Shape shape{1, 5};

    // read value node
    const auto input = make_shared<op::Parameter>(element::f32, shape);
    const auto read_value = make_shared<op::v3::ReadValue>(input, "variable");

    // assign node
    const auto assign = make_shared<op::v3::Assign>(read_value, "variable");

    // ngraph function
    const auto f = make_shared<Function>(assign, ParameterVector{input});

    // auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
}

TEST(op_eval, read_value_dummy)
{
    Shape input_shape{1, 5};

    // read value node
    const auto input = make_shared<op::Parameter>(element::f32, input_shape);
    const auto read_value = make_shared<op::v3::ReadValue>(input, "variable");

    // ngraph function
    const auto f = make_shared<Function>(OutputVector{read_value}, ParameterVector{input});

    vector<float> input_vect = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    vector<float> buffered_vect = {2.f, 2.f, 2.f, 2.f, 2.f};

    HostTensorVector buffer{};
    // auto var = make_host_tensor<element::Type_t::f32>(input_shape, buffered_vect);
    auto var = make_shared<HostTensor>(read_value->get_element_type(),
                                       read_value->get_output_partial_shape(0),
                                       read_value->get_variable_id());
    copy_data(var, buffered_vect);
    buffer.push_back(var);

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(input_shape, input_vect)}, &buffer));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), (Shape{1, 5}));

    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < input_vect.size(); i++)
        EXPECT_NEAR(result_data[i], input_vect[i], 0.000001);
}

TEST(op_eval, read_value_and_assign_dummy)
{
    Shape input_shape{1, 5};

    // read value node
    const auto input = make_shared<op::Parameter>(element::f32, input_shape);
    const auto read_value = make_shared<op::v3::ReadValue>(input, "variable");

    // ngraph function
    const auto f = make_shared<Function>(OutputVector{read_value}, ParameterVector{input});

    vector<float> input_vect = {1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    vector<float> buffered_vect = {2.f, 2.f, 2.f, 2.f, 2.f};

    HostTensorVector buffer{};
    // auto var = make_host_tensor<element::Type_t::f32>(input_shape, buffered_vect);
    // auto var = make_shared<HostTensor>(read_value->get_element_type(),
    //                                   read_value->get_output_partial_shape(0),
    //                                   read_value->get_variable_id());
    // copy_data(var, buffered_vect);
    // buffer.push_back(var);

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(f->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(input_shape, input_vect)}, &buffer));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), (Shape{1, 5}));

    for (auto tensor : buffer)
    {
        if (tensor->get_name() == read_value->get_variable_id())
        {
            cout << "variable " << read_value->get_variable_id() << " found!\n";
            auto buffer_data = read_vector<float>(tensor);
            for (auto i = 0; i < buffer_data.size(); i++)
                cout << buffer_data[i] << " ";
        }
        cout << "\n";
    }

    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < input_vect.size(); i++)
        EXPECT_NEAR(result_data[i], input_vect[i], 0.000001);
}

TEST(op_eval, subgraph_1)
{
    element::Type type = ngraph::element::f32;
    string var_id = "variable_id";

    Shape input_shape{1, 5};
    auto input1 = make_shared<op::Parameter>(type, input_shape);
    auto input2 =
        make_shared<op::Constant>(type, input_shape, std::vector<float>{2.f, 2.f, 2.f, 2.f, 2.f});
    auto read_value = make_shared<op::ReadValue>(input1, var_id);
    auto sum = make_shared<ngraph::op::v1::Add>(read_value, input2);
    auto assign = make_shared<op::Assign>(sum, var_id);

    // ngraph function
    const auto f = make_shared<Function>(OutputVector{assign}, ParameterVector{input1});

    vector<float> input1_vect = {1.f, 1.f, 1.f, 1.f, 1.f};

    HostTensorVector buffer{};
    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(input_shape, input1_vect)}, &buffer));

    for (auto tensor : buffer)
    {
        if (tensor->get_name() == read_value->get_variable_id())
        {
            cout << "variable " << read_value->get_variable_id() << " found!\n";
            auto buffer_data = read_vector<float>(tensor);
            for (auto i = 0; i < buffer_data.size(); i++)
                cout << buffer_data[i] << " ";
        }
        cout << "\n";
    }
}

TEST(op_eval, subgraph_2)
{
    element::Type type = ngraph::element::f32;
    string var_id = "variable_id";

    Shape input_shape{1, 5};
    auto input1 = make_shared<op::Parameter>(type, input_shape);
    auto input2 =
        make_shared<op::Constant>(type, input_shape, std::vector<float>{2.f, 2.f, 2.f, 2.f, 2.f});
    auto read_value = make_shared<op::ReadValue>(input1, var_id);
    auto sum = make_shared<ngraph::op::v1::Add>(read_value, input2);
    auto assign = make_shared<op::Assign>(sum, var_id);

    // ngraph function
    const auto f = make_shared<Function>(OutputVector{assign}, ParameterVector{input1});

    vector<float> input1_vect = {1.f, 1.f, 1.f, 1.f, 1.f};
    vector<float> buffered_vect = {5.f, 5.f, 5.f, 5.f, 5.f};

    HostTensorVector buffer{};
    // auto var = make_host_tensor<element::Type_t::f32>(input_shape, buffered_vect);
    auto var = make_shared<HostTensor>(read_value->get_element_type(),
                                       read_value->get_output_partial_shape(0),
                                       read_value->get_variable_id());
    copy_data(var, buffered_vect);
    buffer.push_back(var);
    auto buffer_data = read_vector<float>(var);
    cout << "initial variable variable_id content:\n";
    for (auto i = 0; i < buffer_data.size(); i++)
        cout << buffer_data[i] << " ";
    cout << "\n";

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate(
        {result}, {make_host_tensor<element::Type_t::f32>(input_shape, input1_vect)}, &buffer));

    for (auto tensor : buffer)
    {
        if (tensor->get_name() == read_value->get_variable_id())
        {
            cout << "variable " << read_value->get_variable_id() << " found!\n";
            auto buffer_data = read_vector<float>(tensor);
            for (auto i = 0; i < buffer_data.size(); i++)
                cout << buffer_data[i] << " ";
        }
        cout << "\n";
    }
}

TEST(op_eval, subgraph_3)
{
    element::Type type = ngraph::element::f32;
    string var_id = "variable_id";

    Shape input_shape{1, 5};
    auto input1 = make_shared<op::Parameter>(type, input_shape);     //   input1    input2
    auto input2 = make_shared<op::Parameter>(type, input_shape);     //      |        |
    auto read_value = make_shared<op::ReadValue>(input1, var_id);    //  read_value   |
    auto sum = make_shared<ngraph::op::v1::Add>(read_value, input2); //      |        |
    auto assign = make_shared<op::Assign>(sum, var_id);              //       \      /
                                                                     //         \  /
                                                                     //         sum
                                                                     //          |
                                                                     //        assign

    // ngraph function
    const auto f = make_shared<Function>(OutputVector{assign}, ParameterVector{input1, input2});

    vector<float> input1_vect = {1.f, 1.f, 1.f, 1.f, 1.f};
    vector<float> input2_vect = {2.f, 2.f, 2.f, 2.f, 2.f};
    vector<float> buffered_vect = {5.f, 5.f, 5.f, 5.f, 5.f};

    HostTensorVector buffer{};
    // auto var = make_host_tensor<element::Type_t::f32>(input_shape, buffered_vect);
    auto var = make_shared<HostTensor>(read_value->get_element_type(),
                                       read_value->get_output_partial_shape(0),
                                       read_value->get_variable_id());
    copy_data(var, buffered_vect);
    buffer.push_back(var);
    auto buffer_data = read_vector<float>(var);
    cout << "initial variable variable_id content:\n";
    for (auto i = 0; i < buffer_data.size(); i++)
        cout << buffer_data[i] << " ";
    cout << "\n";

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(input_shape, input1_vect),
                             make_host_tensor<element::Type_t::f32>(input_shape, input2_vect)},
                            &buffer));

    for (auto tensor : buffer)
    {
        if (tensor->get_name() == read_value->get_variable_id())
        {
            cout << "variable " << read_value->get_variable_id() << " found!\n";
            auto buffer_data = read_vector<float>(tensor);
            for (auto i = 0; i < buffer_data.size(); i++)
                cout << buffer_data[i] << " ";
        }
        cout << "\n";
    }
}