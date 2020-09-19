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

#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>

#include "activation_functions.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/hard_sigmoid.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/tanh.hpp"

using namespace std;
using namespace ngraph;

static shared_ptr<Node> sigmoid(const shared_ptr<Node>& arg, float /* alpha */, float /* beta */)
{
    return make_shared<op::Sigmoid>(arg);
}

static shared_ptr<Node> tanh(const shared_ptr<Node>& arg, float /* alpha */, float /* beta */)
{
    return make_shared<op::Tanh>(arg);
}

static shared_ptr<Node> relu(const shared_ptr<Node>& arg, float /* alpha */, float /* beta */)
{
    return make_shared<op::Relu>(arg);
}

static shared_ptr<Node> hardsigmoid(const shared_ptr<Node>& arg, float alpha, float beta)
{
    const auto alpha_node = op::Constant::create<float>(arg->get_element_type(), Shape{}, {alpha});
    const auto beta_node = op::Constant::create<float>(arg->get_element_type(), Shape{}, {beta});

    return make_shared<op::HardSigmoid>(arg, alpha_node, beta_node);
}

op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f, float alpha, float beta)
    : m_function{f}
    , m_alpha{alpha}
    , m_beta{beta}
{
}

op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f, float alpha)
    : ActivationFunction(f, alpha, nanf(""))
{
}

op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f)
    : ActivationFunction(f, nanf(""), nanf(""))
{
}

shared_ptr<Node> op::util::ActivationFunction::operator()(const shared_ptr<Node>& arg) const
{
    return m_function(arg, m_alpha, m_beta);
}

op::util::ActivationFunction op::util::get_activation_func_by_name(const string& func_name)
{
    using ActivationFunctionMap = unordered_map<string, op::util::ActivationFunction>;

    static ActivationFunctionMap func_map{
        {"sigmoid", op::util::ActivationFunction{sigmoid}},
        {"tanh", op::util::ActivationFunction{tanh}},
        {"relu", op::util::ActivationFunction{relu}},
        {"hardsigmoid", op::util::ActivationFunction{hardsigmoid, 0.2f, 0.5f}},
    };

    auto func_it = func_map.find(func_name);
    if (func_it == end(func_map))
    {
        throw op::util::error::UnknownActivationFunction(func_name);
    }
    return func_it->second;
}
