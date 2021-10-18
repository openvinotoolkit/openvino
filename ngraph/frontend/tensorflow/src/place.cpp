// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend_manager/frontend_exceptions.hpp>

#include "place.hpp"
#include "node_context.hpp"
#include "op_def.pb.h"
#include "tensor.pb.h"
#include "types.pb.h"

namespace ov {
namespace frontend {

bool PlaceTF::is_input() const {
    const auto& model_ins = m_input_model.get_inputs();

    const auto cmp = [this](const ngraph::frontend::Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool PlaceTF::is_output() const {
    const auto& model_outs = m_input_model.get_outputs();
    const auto cmp = [this](const ngraph::frontend::Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}

OpPlaceTF::OpPlaceTF(const ngraph::frontend::InputModel& input_model, std::shared_ptr<DecoderBase> op_decoder)
    : PlaceTF(input_model, {op_decoder->get_op_name()}),
      m_op_decoder(op_decoder) {}

const std::vector<std::shared_ptr<OutPortPlaceTF>>& OpPlaceTF::get_output_ports() const {
    return m_output_ports;
}

const std::map<std::string, std::vector<std::shared_ptr<InPortPlaceTF>>>& OpPlaceTF::get_input_ports() const {
    return m_input_ports;
}

std::shared_ptr<InPortPlaceTF> OpPlaceTF::get_input_port_tf(const std::string& inputName, int inputPortIndex) const {
    FRONT_END_GENERAL_CHECK(inputPortIndex <= m_input_ports.at(inputName).size(), "inputPortIndex is out of bounds.");
    return m_input_ports.at(inputName)[inputPortIndex];
}

std::shared_ptr<DecoderBase> OpPlaceTF::get_decoder() const {
    return m_op_decoder;
}

void OpPlaceTF::add_out_port(const std::shared_ptr<OutPortPlaceTF>& output, int idx) {
    while (idx >= m_output_ports.size()) {
        m_output_ports.push_back(std::shared_ptr<OutPortPlaceTF>());
    }
    m_output_ports[idx] = output;
}

void OpPlaceTF::add_in_port(const std::shared_ptr<InPortPlaceTF>& input, const std::string& name) {
    m_input_ports[name].push_back(input);
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_input_port(const std::string& name) const {
    FRONT_END_GENERAL_CHECK(m_input_ports.at(name).size() == 1, "Only one input port should exist.");
    return m_input_ports.at(name)[0];
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_input_port(int outputPortIndex) const {
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1, "Only one named input port should exist.");
    return m_input_ports.begin()->second[outputPortIndex];
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_output_port(int outputPortIndex) const {
    FRONT_END_GENERAL_CHECK(m_output_ports.size() > outputPortIndex, "No port with index: ", outputPortIndex);
    return m_output_ports[outputPortIndex];
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_output_port() const {
    FRONT_END_GENERAL_CHECK(m_output_ports.size() == 1, "Only one output port should exist.");
    return m_output_ports[0];
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_input_port() const {
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1 && m_input_ports.begin()->second.size() == 1,
                            "Only one input port should exist.");
    return m_input_ports.begin()->second[0];
}

std::vector<ngraph::frontend::Place::Ptr> OpPlaceTF::get_consuming_operations() const {
    std::vector<ngraph::frontend::Place::Ptr> consuming_ops;
    for (const auto& out_port : m_output_ports) {
        auto consuming_ops_out = out_port->get_consuming_operations();
        consuming_ops.insert(consuming_ops.end(), consuming_ops_out.begin(), consuming_ops_out.end());
    }
    return consuming_ops;
}

std::vector<ngraph::frontend::Place::Ptr> OpPlaceTF::get_consuming_operations(int outputPortIndex) const {
    return get_output_port(outputPortIndex)->get_consuming_operations();
}

std::vector<ngraph::frontend::Place::Ptr> OpPlaceTF::get_consuming_ports() const {
    std::vector<ngraph::frontend::Place::Ptr> consuming_ports;
    for (const auto& out_port : m_output_ports) {
        auto consuming_ops_out = out_port->get_consuming_ports();
        consuming_ports.insert(consuming_ports.end(), consuming_ops_out.begin(), consuming_ops_out.end());
    }
    return consuming_ports;
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_input_port(const std::string& inputName, int inputPortIndex) const {
    FRONT_END_GENERAL_CHECK(inputPortIndex <= m_input_ports.at(inputName).size(), "inputPortIndex is out of bounds.");
    return m_input_ports.at(inputName)[inputPortIndex];
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_source_tensor() const {
    return get_input_port()->get_source_tensor();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_source_tensor(const std::string& inputName) const {
    return get_input_port(inputName)->get_source_tensor();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_source_tensor(int inputPortIndex) const {
    return get_input_port(inputPortIndex)->get_source_tensor();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return get_input_port(inputName, inputPortIndex)->get_source_tensor();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_target_tensor() const {
    return get_output_port()->get_target_tensor();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_producing_operation(const std::string& inputName) const {
    return get_input_port(inputName)->get_producing_operation();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_producing_operation(const std::string& inputName,
                                                                int inputPortIndex) const {
    return get_input_port(inputName, inputPortIndex)->get_producing_operation();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_producing_operation() const {
    return get_input_port()->get_producing_operation();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_producing_operation(int inputPortIndex) const {
    return get_input_port(inputPortIndex)->get_producing_operation();
}

ngraph::frontend::Place::Ptr OpPlaceTF::get_target_tensor(int outputPortIndex) const {
    return get_output_port(outputPortIndex)->get_target_tensor();
}

TensorPlaceTF::TensorPlaceTF(const ngraph::frontend::InputModel& input_model,
                             const ov::PartialShape& pshape,
                             ov::element::Type type,
                             const std::vector<std::string>& names)
    : PlaceTF(input_model, names),
      m_pshape(pshape),
      m_type(type) {}

std::vector<ngraph::frontend::Place::Ptr> TensorPlaceTF::get_consuming_ports() const {
    std::vector<ngraph::frontend::Place::Ptr> consuming_ports;
    for (const auto& consuming_port : m_consuming_ports) {
        if (const auto& locked = consuming_port.lock()) {
            consuming_ports.push_back(locked);
        } else {
            FRONT_END_THROW("Consuming Port has expired.");
        }
    }
    return consuming_ports;
}

ngraph::frontend::Place::Ptr TensorPlaceTF::get_producing_port() const {
    FRONT_END_GENERAL_CHECK(m_producing_ports.size() == 1, "Only one producing port is supported.");
    if (const auto& producing_port = m_producing_ports[0].lock()) {
        return producing_port;
    }
    FRONT_END_THROW("Producing Port has expired.");
}

void TensorPlaceTF::add_producing_port(const std::shared_ptr<OutPortPlaceTF>& out_port) {
    m_producing_ports.push_back(out_port);
}

void TensorPlaceTF::add_consuming_port(const std::shared_ptr<InPortPlaceTF>& in_port) {
    m_consuming_ports.push_back(in_port);
}

std::vector<ngraph::frontend::Place::Ptr> TensorPlaceTF::get_consuming_operations() const {
    std::vector<ngraph::frontend::Place::Ptr> consuming_ops;
    for (const auto& consuming_port : m_consuming_ports) {
        if (auto port_ptr = consuming_port.lock()) {
            auto port_consuming_ops = port_ptr->get_consuming_operations();
            consuming_ops.insert(consuming_ops.end(), port_consuming_ops.begin(), port_consuming_ops.end());
        } else {
            FRONT_END_THROW("Port has expired.");
        }
    }
    return consuming_ops;
}

bool TensorPlaceTF::is_equal_data(ngraph::frontend::Place::Ptr another) const {
    auto consuming_ports = get_consuming_ports();
    bool eq_to_consuming_port =
        std::any_of(consuming_ports.begin(), consuming_ports.end(), [&another](const Ptr& place) {
            return place->is_equal(another);
        });
    return is_equal(another) || get_producing_port()->is_equal(another) || eq_to_consuming_port;
}

ngraph::frontend::Place::Ptr TensorPlaceTF::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<TensorPlaceTF> InPortPlaceTF::get_source_tensor_tf() const {
    if (const auto& tensor = m_source_tensor.lock()) {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

std::shared_ptr<OpPlaceTF> InPortPlaceTF::get_op() {
    if (const auto& op = m_op.lock()) {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}

void InPortPlaceTF::set_source_tensor(const std::weak_ptr<TensorPlaceTF>& source_tensor) {
    m_source_tensor = source_tensor;
}

std::vector<ngraph::frontend::Place::Ptr> InPortPlaceTF::get_consuming_operations() const {
    if (const auto& consuming_op = m_op.lock()) {
        return {consuming_op};
    }
    FRONT_END_THROW("Operation has expired.");
}

ngraph::frontend::Place::Ptr InPortPlaceTF::get_source_tensor() const {
    if (const auto& tensor = m_source_tensor.lock()) {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

ngraph::frontend::Place::Ptr InPortPlaceTF::get_producing_port() const {
    return get_source_tensor()->get_producing_port();
}

bool InPortPlaceTF::is_equal_data(ngraph::frontend::Place::Ptr another) const {
    return get_source_tensor()->is_equal_data(another);
}

ngraph::frontend::Place::Ptr InPortPlaceTF::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<TensorPlaceTF> OutPortPlaceTF::get_target_tensor_tf() const {
    if (const auto& target_tensor = m_target_tensor.lock()) {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

std::vector<ngraph::frontend::Place::Ptr> OutPortPlaceTF::get_consuming_operations() const {
    if (auto tensor_ptr = m_target_tensor.lock()) {
        return tensor_ptr->get_consuming_operations();
    }
    FRONT_END_THROW("Tensor has expired.");
}

void OutPortPlaceTF::set_target_tensor(const std::weak_ptr<TensorPlaceTF>& target_tensor) {
    m_target_tensor = target_tensor;
}

std::vector<ngraph::frontend::Place::Ptr> OutPortPlaceTF::get_consuming_ports() const {
    if (auto tensor_ptr = m_target_tensor.lock()) {
        return tensor_ptr->get_consuming_ports();
    }
    FRONT_END_THROW("Tensor has expired.");
}

bool OutPortPlaceTF::is_equal_data(ngraph::frontend::Place::Ptr another) const {
    return get_target_tensor()->is_equal_data(another);
}

ngraph::frontend::Place::Ptr OutPortPlaceTF::get_target_tensor() const {
    if (const auto& target_tensor = m_target_tensor.lock()) {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

ngraph::frontend::Place::Ptr OutPortPlaceTF::get_producing_operation() const {
    if (auto op = m_op.lock()) {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}
}  // namespace frontend
}  // namespace ov