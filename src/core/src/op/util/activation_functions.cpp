// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/activation_functions.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/hard_sigmoid.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/tanh.hpp"

using namespace std;

static shared_ptr<ov::Node> sigmoid(const shared_ptr<ov::Node>& arg, float /* alpha */, float /* beta */) {
    return make_shared<ngraph::op::Sigmoid>(arg);
}

static shared_ptr<ov::Node> tanh(const shared_ptr<ov::Node>& arg, float /* alpha */, float /* beta */) {
    return make_shared<ngraph::op::Tanh>(arg);
}

static shared_ptr<ov::Node> relu(const shared_ptr<ov::Node>& arg, float /* alpha */, float /* beta */) {
    return make_shared<ngraph::op::Relu>(arg);
}

static shared_ptr<ov::Node> hardsigmoid(const shared_ptr<ov::Node>& arg, float alpha, float beta) {
    const auto alpha_node = ngraph::op::Constant::create<float>(arg->get_element_type(), ngraph::Shape{}, {alpha});
    const auto beta_node = ngraph::op::Constant::create<float>(arg->get_element_type(), ngraph::Shape{}, {beta});

    return make_shared<ngraph::op::HardSigmoid>(arg, alpha_node, beta_node);
}

ov::op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f, float alpha, float beta)
    : m_function{f},
      m_alpha{alpha},
      m_beta{beta} {}

ov::op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f, float alpha)
    : ActivationFunction(f, alpha, nanf("")) {}

ov::op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f)
    : ActivationFunction(f, nanf(""), nanf("")) {}

shared_ptr<ov::Node> ov::op::util::ActivationFunction::operator()(const shared_ptr<Node>& arg) const {
    return m_function(arg, m_alpha, m_beta);
}

ov::op::util::ActivationFunction ov::op::util::get_activation_func_by_name(const string& func_name) {
    using ActivationFunctionMap = unordered_map<string, op::util::ActivationFunction>;

    static ActivationFunctionMap func_map{
        {"sigmoid", op::util::ActivationFunction{sigmoid}},
        {"tanh", op::util::ActivationFunction{tanh}},
        {"relu", op::util::ActivationFunction{relu}},
        {"hardsigmoid", op::util::ActivationFunction{hardsigmoid, 0.2f, 0.5f}},
    };

    auto func_it = func_map.find(func_name);
    if (func_it == end(func_map)) {
        throw op::util::error::UnknownActivationFunction(func_name);
    }
    return func_it->second;
}
