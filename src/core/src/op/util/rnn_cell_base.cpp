// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/rnn_cell_base.hpp"

#include <algorithm>
#include <iterator>
#include <locale>

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/util/common_util.hpp"

std::shared_ptr<ov::Node> ov::op::util::convert_lstm_node_format(const Output<Node>& node,
                                                                 LSTMWeightsFormat from_format,
                                                                 LSTMWeightsFormat to_format,
                                                                 int64_t axis) {
    static const std::map<op::util::LSTMWeightsFormat, std::vector<size_t>> gate_order_map{
        {op::util::LSTMWeightsFormat::FICO, {0, 1, 2, 3}},
        {op::util::LSTMWeightsFormat::ICOF, {1, 2, 3, 0}},
        {op::util::LSTMWeightsFormat::IFOC, {1, 0, 3, 2}},
        {op::util::LSTMWeightsFormat::IOFC, {1, 3, 0, 2}},
        {op::util::LSTMWeightsFormat::IFCO, {1, 0, 2, 3}},
    };
    const auto& from = gate_order_map.at(from_format);
    const auto& to = gate_order_map.at(to_format);
    size_t num_gates = 4;

    auto axis_const = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{}, axis);
    OutputVector splitted_node = std::make_shared<ov::op::v1::Split>(node, axis_const, num_gates)->outputs();
    OutputVector nodes_in_new_format(num_gates);
    for (size_t i = 0; i < num_gates; ++i) {
        nodes_in_new_format[to[from[i]]] = splitted_node[i];
    }
    return std::make_shared<ov::op::v0::Concat>(nodes_in_new_format, axis);
}

std::shared_ptr<ov::Node> ov::op::util::convert_lstm_peepholes_format(const Output<Node>& node,
                                                                      LSTMPeepholesFormat from_format,
                                                                      LSTMPeepholesFormat to_format,
                                                                      int64_t axis) {
    static const std::map<op::util::LSTMPeepholesFormat, std::vector<size_t>> gate_order_map{
        {op::util::LSTMPeepholesFormat::FIO, {0, 1, 2}},
        {op::util::LSTMPeepholesFormat::IFO, {1, 0, 2}},
        {op::util::LSTMPeepholesFormat::IOF, {1, 2, 0}},
    };
    const auto& from = gate_order_map.at(from_format);
    const auto& to = gate_order_map.at(to_format);
    size_t num_gates = 3;

    auto axis_const = std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape{}, axis);
    OutputVector splitted_node = std::make_shared<ov::op::v1::Split>(node, axis_const, num_gates)->outputs();
    OutputVector nodes_in_new_format(num_gates);
    for (size_t i = 0; i < num_gates; ++i) {
        nodes_in_new_format[to[from[i]]] = splitted_node[i];
    }
    return std::make_shared<ov::op::v0::Concat>(nodes_in_new_format, axis);
}

// Modify input vector in-place and return reference to modified vector.
static std::vector<std::string> to_lower_case(const std::vector<std::string>& vs) {
    std::vector<std::string> res(vs);
    transform(begin(res), end(res), begin(res), [](std::string& s) {
        return ov::util::to_lower(s);
    });
    return res;
}

ov::op::util::RNNCellBase::RNNCellBase() : m_hidden_size(0), m_clip(0.f) {}

ov::op::util::RNNCellBase::RNNCellBase(const OutputVector& args,
                                       size_t hidden_size,
                                       float clip,
                                       const std::vector<std::string>& activations,
                                       const std::vector<float>& activations_alpha,
                                       const std::vector<float>& activations_beta)
    : Op(args),
      m_hidden_size(hidden_size),
      m_clip(clip),
      m_activations(to_lower_case(activations)),
      m_activations_alpha(activations_alpha),
      m_activations_beta(activations_beta) {}

bool ov::op::util::RNNCellBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_RNNCellBase_visit_attributes);
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("activations", m_activations);
    visitor.on_attribute("activations_alpha", m_activations_alpha);
    visitor.on_attribute("activations_beta", m_activations_beta);
    visitor.on_attribute("clip", m_clip);
    return true;
}

void ov::op::util::RNNCellBase::validate_input_rank_dimension(const std::vector<ov::PartialShape>& input) {
    enum { X, initial_hidden_state, W, R, B };

    // Verify static ranks for all inputs
    for (size_t i = 0; i < input.size(); i++) {
        NODE_VALIDATION_CHECK(this,
                              (input[i].rank().is_static()),
                              "RNNCellBase supports only static rank for input tensors. Input ",
                              i);
    }

    // Verify input dimension against values provided in spec (LSTMCell_1.md)
    for (size_t i = 0; i < input.size(); i++) {
        if (i == B) {
            // verify only B input dimension which is 1D
            NODE_VALIDATION_CHECK(this,
                                  (input[i].rank().get_length() == 1),
                                  "RNNCellBase B input tensor dimension is not correct.");
        } else {
            // Verify all other input dimensions which are 2D tensor types
            NODE_VALIDATION_CHECK(this,
                                  (input[i].rank().get_length() == 2),
                                  "RNNCellBase input tensor dimension is not correct for ",
                                  i,
                                  " input parameter. Current input length: ",
                                  input[i].rank().get_length(),
                                  ", expected: 2.");
        }
    }

    // Compare input_size dimension for X and W inputs
    const auto& x_pshape = input.at(X);
    const auto& w_pshape = input.at(W);

    NODE_VALIDATION_CHECK(this, (x_pshape[1].compatible(w_pshape[1])), "RNNCellBase mismatched input_size dimension.");
}

ov::op::util::ActivationFunction ov::op::util::RNNCellBase::get_activation_function(size_t idx) const {
    // Normalize activation function case.
    std::string func_name = m_activations.at(idx);
    std::locale loc;
    std::transform(func_name.begin(), func_name.end(), func_name.begin(), [&loc](char c) {
        return std::tolower(c, loc);
    });

    op::util::ActivationFunction afunc = get_activation_func_by_name(func_name);

    // Set activation functions parameters (if any)
    if (m_activations_alpha.size() > idx) {
        afunc.set_alpha(m_activations_alpha.at(idx));
    }
    if (m_activations_beta.size() > idx) {
        afunc.set_beta(m_activations_beta.at(idx));
    }

    return afunc;
}

std::shared_ptr<ov::Node> ov::op::util::RNNCellBase::add(const Output<Node>& lhs, const Output<Node>& rhs) {
    return {std::make_shared<ov::op::v1::Add>(lhs, rhs)};
}

std::shared_ptr<ov::Node> ov::op::util::RNNCellBase::sub(const Output<Node>& lhs, const Output<Node>& rhs) {
    return {std::make_shared<ov::op::v1::Subtract>(lhs, rhs)};
}

std::shared_ptr<ov::Node> ov::op::util::RNNCellBase::mul(const Output<Node>& lhs, const Output<Node>& rhs) {
    return {std::make_shared<ov::op::v1::Multiply>(lhs, rhs)};
}

std::shared_ptr<ov::Node> ov::op::util::RNNCellBase::clip(const Output<Node>& data) const {
    if (m_clip == 0.f) {
        return data.get_node_shared_ptr();
    }

    return std::make_shared<ov::op::v0::Clamp>(data, -m_clip, m_clip);
}
