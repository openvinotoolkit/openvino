// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/activation_functions.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>

#include "openvino/op/constant.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tanh.hpp"

static std::shared_ptr<ov::Node> sigmoid(const std::shared_ptr<ov::Node>& arg, float /* alpha */, float /* beta */) {
    return std::make_shared<ov::op::v0::Sigmoid>(arg);
}

static std::shared_ptr<ov::Node> tanh(const std::shared_ptr<ov::Node>& arg, float /* alpha */, float /* beta */) {
    return std::make_shared<ov::op::v0::Tanh>(arg);
}

static std::shared_ptr<ov::Node> relu(const std::shared_ptr<ov::Node>& arg, float /* alpha */, float /* beta */) {
    return std::make_shared<ov::op::v0::Relu>(arg);
}

static std::shared_ptr<ov::Node> hardsigmoid(const std::shared_ptr<ov::Node>& arg, float alpha, float beta) {
    const auto alpha_node = ov::op::v0::Constant::create<float>(arg->get_element_type(), ov::Shape{}, {alpha});
    const auto beta_node = ov::op::v0::Constant::create<float>(arg->get_element_type(), ov::Shape{}, {beta});

    return std::make_shared<ov::op::v0::HardSigmoid>(arg, alpha_node, beta_node);
}

ov::op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f, float alpha, float beta)
    : m_function{f},
      m_alpha{alpha},
      m_beta{beta} {}

ov::op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f, float alpha)
    : ActivationFunction(f, alpha, nanf("")) {}

ov::op::util::ActivationFunction::ActivationFunction(ActivationFunctionType f)
    : ActivationFunction(f, nanf(""), nanf("")) {}

std::shared_ptr<ov::Node> ov::op::util::ActivationFunction::operator()(const std::shared_ptr<Node>& arg) const {
    return m_function(arg, m_alpha, m_beta);
}

ov::op::util::ActivationFunction ov::op::util::get_activation_func_by_name(const std::string& func_name) {
    using ActivationFunctionMap = std::unordered_map<std::string, op::util::ActivationFunction>;

    static ActivationFunctionMap func_map{
        {"sigmoid", op::util::ActivationFunction{::sigmoid}},
        {"tanh", op::util::ActivationFunction{::tanh}},
        {"relu", op::util::ActivationFunction{::relu}},
        {"hardsigmoid", op::util::ActivationFunction{::hardsigmoid, 0.2f, 0.5f}},
    };

    auto func_it = func_map.find(func_name);
    if (func_it == end(func_map)) {
        throw op::util::error::UnknownActivationFunction(func_name);
    }
    return func_it->second;
}
