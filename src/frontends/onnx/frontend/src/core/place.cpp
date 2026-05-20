// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace onnx {
bool Place::is_input() const {
    const auto& model_ins = m_input_model.get_inputs();

    const auto cmp = [this](const ov::frontend::Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool Place::is_output() const {
    const auto& model_outs = m_input_model.get_outputs();
    const auto cmp = [this](const ov::frontend::Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}

OpPlace::OpPlace(const ov::frontend::InputModel& input_model, std::shared_ptr<DecoderBase> op_decoder)
    : Place(input_model, {op_decoder->get_op_name()}),
      m_op_decoder(op_decoder),
      m_back_edge_set(false) {}

void OpPlace::bind_input_tensor(const std::shared_ptr<TensorPlace>& tensor, const std::string& port_name) {
    m_input_bindings.push_back(InputBinding{tensor, port_name});
}

void OpPlace::bind_output_tensor(const std::shared_ptr<TensorPlace>& tensor, int idx) {
    FRONT_END_GENERAL_CHECK(idx >= 0, "Output port index to be bound is negative.");
    const auto out_idx = static_cast<size_t>(idx);
    if (out_idx >= m_output_bindings.size()) {
        m_output_bindings.resize(out_idx + 1);
    }
    m_output_bindings[out_idx] = tensor;
}

void OpPlace::ensure_ports_materialized() const {
    if (m_ports_materialized) {
        return;
    }
    // Bindings (op<->tensor connectivity) are populated lazily by the owning InputModel
    // on first port API access. If a provider is registered, run it before materializing
    // the InPort/OutPort objects so they have something to wire.
    if (auto provider = lazy_bindings_provider()) {
        provider->ensure_bindings_populated();
    }
    m_ports_materialized = true;

    // const_cast is safe: shared_from_this() is logically const and we only mutate
    // members declared mutable (m_input_ports / m_output_ports).
    auto self = const_cast<OpPlace*>(this)->shared_from_this();
    const auto& input_model = get_input_model_ref();

    for (const auto& binding : m_input_bindings) {
        auto tensor = binding.tensor.lock();
        if (!tensor) {
            continue;
        }
        auto in_port = std::make_shared<InPortPlace>(input_model);
        in_port->set_op(self);
        in_port->set_source_tensor(tensor);
        tensor->add_consuming_port(in_port);
        m_input_ports[binding.port_name].push_back(in_port);
    }
    for (size_t i = 0; i < m_output_bindings.size(); ++i) {
        auto tensor = m_output_bindings[i].lock();
        if (!tensor) {
            continue;
        }
        auto out_port = std::make_shared<OutPortPlace>(input_model);
        out_port->set_op(self);
        out_port->set_target_tensor(tensor);
        tensor->add_producing_port(out_port);
        if (i >= m_output_ports.size()) {
            m_output_ports.resize(i + 1);
        }
        m_output_ports[i] = out_port;
    }
}

void OpPlace::set_next_iteration_back_edge(const std::string& next_iteration_producer_name,
                                           size_t next_iteration_producer_output_port_idx) {
    m_next_iteration_producer_name = next_iteration_producer_name;
    m_next_iteration_producer_output_port_idx = next_iteration_producer_output_port_idx;
    m_back_edge_set = true;
}

void OpPlace::get_next_iteration_back_edge(std::string& next_iteration_producer_name,
                                           size_t& next_iteration_producer_output_port_idx) const {
    FRONT_END_GENERAL_CHECK(m_back_edge_set, "[ONNX Frontend] internal error: back edge for NextIteration is not set");
    next_iteration_producer_name = m_next_iteration_producer_name;
    next_iteration_producer_output_port_idx = m_next_iteration_producer_output_port_idx;
}

const std::vector<std::shared_ptr<OutPortPlace>>& OpPlace::get_output_ports() const {
    ensure_ports_materialized();
    return m_output_ports;
}

const std::map<std::string, std::vector<std::shared_ptr<InPortPlace>>>& OpPlace::get_input_ports() const {
    ensure_ports_materialized();
    return m_input_ports;
}

std::shared_ptr<DecoderBase> OpPlace::get_decoder() const {
    return m_op_decoder;
}

void OpPlace::add_out_port(const std::shared_ptr<OutPortPlace>& output, int idx) {
    FRONT_END_GENERAL_CHECK(idx >= 0, "Output port index to be added is negative.");
    size_t output_port_index = static_cast<size_t>(idx);
    ensure_ports_materialized();
    while (output_port_index >= m_output_ports.size()) {
        m_output_ports.push_back(std::shared_ptr<OutPortPlace>());
    }
    m_output_ports[output_port_index] = output;
}

void OpPlace::add_in_port(const std::shared_ptr<InPortPlace>& input, const std::string& name) {
    ensure_ports_materialized();
    m_input_ports[name].push_back(input);
}

ov::frontend::Place::Ptr OpPlace::get_input_port(const std::string& name) const {
    ensure_ports_materialized();
    FRONT_END_GENERAL_CHECK(m_input_ports.at(name).size() == 1, "Only one input port should exist.");
    return m_input_ports.at(name)[0];
}

ov::frontend::Place::Ptr OpPlace::get_input_port(int outputPortIndex) const {
    ensure_ports_materialized();
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1, "Only one named input port should exist.");
    return m_input_ports.begin()->second[outputPortIndex];
}

ov::frontend::Place::Ptr OpPlace::get_output_port(int outputPortIndex) const {
    ensure_ports_materialized();
    FRONT_END_GENERAL_CHECK(outputPortIndex >= 0, "outputPortIndex is negative.");
    size_t output_port_index = static_cast<size_t>(outputPortIndex);
    FRONT_END_GENERAL_CHECK(m_output_ports.size() > output_port_index, "No port with index: ", output_port_index);
    return m_output_ports[output_port_index];
}

ov::frontend::Place::Ptr OpPlace::get_output_port() const {
    ensure_ports_materialized();
    FRONT_END_GENERAL_CHECK(m_output_ports.size() == 1, "Only one output port should exist.");
    return m_output_ports[0];
}

ov::frontend::Place::Ptr OpPlace::get_input_port() const {
    ensure_ports_materialized();
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1 && m_input_ports.begin()->second.size() == 1,
                            "Only one input port should exist.");
    return m_input_ports.begin()->second[0];
}

std::vector<ov::frontend::Place::Ptr> OpPlace::get_consuming_operations() const {
    ensure_ports_materialized();
    std::vector<ov::frontend::Place::Ptr> consuming_ops;
    for (const auto& out_port : m_output_ports) {
        auto consuming_ops_out = out_port->get_consuming_operations();
        consuming_ops.insert(consuming_ops.end(), consuming_ops_out.begin(), consuming_ops_out.end());
    }
    return consuming_ops;
}

std::vector<ov::frontend::Place::Ptr> OpPlace::get_consuming_operations(int outputPortIndex) const {
    return get_output_port(outputPortIndex)->get_consuming_operations();
}

std::vector<ov::frontend::Place::Ptr> OpPlace::get_consuming_ports() const {
    ensure_ports_materialized();
    std::vector<ov::frontend::Place::Ptr> consuming_ports;
    for (const auto& out_port : m_output_ports) {
        auto consuming_ops_out = out_port->get_consuming_ports();
        consuming_ports.insert(consuming_ports.end(), consuming_ops_out.begin(), consuming_ops_out.end());
    }
    return consuming_ports;
}

ov::frontend::Place::Ptr OpPlace::get_input_port(const std::string& inputName, int inputPortIndex) const {
    ensure_ports_materialized();
    FRONT_END_GENERAL_CHECK(inputPortIndex >= 0, "inputPortIndex is negative.");
    size_t input_port_index = static_cast<size_t>(inputPortIndex);
    FRONT_END_GENERAL_CHECK(input_port_index <= m_input_ports.at(inputName).size(), "inputPortIndex is out of bounds.");
    return m_input_ports.at(inputName)[input_port_index];
}

ov::frontend::Place::Ptr OpPlace::get_source_tensor() const {
    return get_input_port()->get_source_tensor();
}

ov::frontend::Place::Ptr OpPlace::get_source_tensor(const std::string& inputName) const {
    return get_input_port(inputName)->get_source_tensor();
}

ov::frontend::Place::Ptr OpPlace::get_source_tensor(int inputPortIndex) const {
    return get_input_port(inputPortIndex)->get_source_tensor();
}

ov::frontend::Place::Ptr OpPlace::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return get_input_port(inputName, inputPortIndex)->get_source_tensor();
}

ov::frontend::Place::Ptr OpPlace::get_target_tensor() const {
    return get_output_port()->get_target_tensor();
}

ov::frontend::Place::Ptr OpPlace::get_producing_operation(const std::string& inputName) const {
    return get_input_port(inputName)->get_producing_operation();
}

ov::frontend::Place::Ptr OpPlace::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    return get_input_port(inputName, inputPortIndex)->get_producing_operation();
}

ov::frontend::Place::Ptr OpPlace::get_producing_operation() const {
    return get_input_port()->get_producing_operation();
}

ov::frontend::Place::Ptr OpPlace::get_producing_operation(int inputPortIndex) const {
    return get_input_port(inputPortIndex)->get_producing_operation();
}

ov::frontend::Place::Ptr OpPlace::get_target_tensor(int outputPortIndex) const {
    return get_output_port(outputPortIndex)->get_target_tensor();
}

TensorPlace::TensorPlace(const ov::frontend::InputModel& input_model,
                         const ov::PartialShape& pshape,
                         ov::element::Type type,
                         const std::vector<std::string>& names)
    : Place(input_model, names),
      m_pshape(pshape),
      m_type(type) {
    m_operation_name = (names.size() > 0) ? names[0] : m_operation_name;
}

TensorPlace::TensorPlace(const ov::frontend::InputModel& input_model,
                         const ov::PartialShape& pshape,
                         ov::element::Type type,
                         const std::vector<std::string>& names,
                         const std::string& operation_name)
    : Place(input_model, names),
      m_pshape(pshape),
      m_type(type),
      m_operation_name(operation_name) {}

std::vector<ov::frontend::Place::Ptr> TensorPlace::get_consuming_ports() const {
    ensure_consuming_ports_materialized();
    std::vector<ov::frontend::Place::Ptr> consuming_ports;
    for (const auto& consuming_port : m_consuming_ports) {
        if (const auto& locked = consuming_port.lock()) {
            consuming_ports.push_back(locked);
        } else {
            FRONT_END_THROW("Consuming Port has expired.");
        }
    }
    return consuming_ports;
}

ov::frontend::Place::Ptr TensorPlace::get_producing_port() const {
    ensure_producing_port_materialized();
    FRONT_END_GENERAL_CHECK(m_producing_ports.size() == 1, "Only one producing port is supported.");
    if (const auto& producing_port = m_producing_ports[0].lock()) {
        return producing_port;
    }
    FRONT_END_THROW("Producing Port has expired.");
}

void TensorPlace::add_producing_port(const std::shared_ptr<OutPortPlace>& out_port) {
    m_producing_ports.push_back(out_port);
}

void TensorPlace::add_consuming_port(const std::shared_ptr<InPortPlace>& in_port) {
    m_consuming_ports.push_back(in_port);
}

void TensorPlace::bind_producing_op(const std::shared_ptr<OpPlace>& op, int out_idx) {
    m_producing_op = op;
    m_producing_op_out_idx = out_idx;
}

void TensorPlace::bind_consuming_op(const std::shared_ptr<OpPlace>& op, const std::string& port_name) {
    m_consuming_op_bindings.emplace_back(op, port_name);
}

void TensorPlace::ensure_producing_port_materialized() const {
    if (m_producing_port_materialized) {
        return;
    }
    if (auto provider = lazy_bindings_provider()) {
        provider->ensure_bindings_populated();
    }
    m_producing_port_materialized = true;
    if (auto op = m_producing_op.lock()) {
        // Triggers materialization of all of op's input/output ports, which in turn pushes
        // an OutPortPlace onto this tensor's m_producing_ports.
        op->ensure_ports_materialized();
    }
}

void TensorPlace::ensure_consuming_ports_materialized() const {
    if (m_consuming_ports_materialized) {
        return;
    }
    if (auto provider = lazy_bindings_provider()) {
        provider->ensure_bindings_populated();
    }
    m_consuming_ports_materialized = true;
    for (const auto& binding : m_consuming_op_bindings) {
        if (auto op = binding.first.lock()) {
            op->ensure_ports_materialized();
        }
    }
}

std::vector<ov::frontend::Place::Ptr> TensorPlace::get_consuming_operations() const {
    ensure_consuming_ports_materialized();
    std::vector<ov::frontend::Place::Ptr> consuming_ops;
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

bool TensorPlace::is_equal_data(const ov::frontend::Place::Ptr& another) const {
    auto consuming_ports = get_consuming_ports();
    bool eq_to_consuming_port =
        std::any_of(consuming_ports.begin(), consuming_ports.end(), [&another](const Ptr& place) {
            return place->is_equal(another);
        });
    return is_equal(another) || get_producing_port()->is_equal(another) || eq_to_consuming_port;
}

ov::frontend::Place::Ptr TensorPlace::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<OpPlace> InPortPlace::get_op() {
    if (const auto& op = m_op.lock()) {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}

void InPortPlace::set_source_tensor(const std::weak_ptr<TensorPlace>& source_tensor) {
    m_source_tensor = source_tensor;
}

std::vector<ov::frontend::Place::Ptr> InPortPlace::get_consuming_operations() const {
    if (const auto& consuming_op = m_op.lock()) {
        return {consuming_op};
    }
    FRONT_END_THROW("Operation has expired.");
}

ov::frontend::Place::Ptr InPortPlace::get_source_tensor() const {
    if (const auto& tensor = m_source_tensor.lock()) {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

ov::frontend::Place::Ptr InPortPlace::get_producing_port() const {
    return get_source_tensor()->get_producing_port();
}

bool InPortPlace::is_equal_data(const ov::frontend::Place::Ptr& another) const {
    return get_source_tensor()->is_equal_data(another);
}

ov::frontend::Place::Ptr InPortPlace::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

std::vector<ov::frontend::Place::Ptr> OutPortPlace::get_consuming_operations() const {
    if (auto tensor_ptr = m_target_tensor.lock()) {
        return tensor_ptr->get_consuming_operations();
    }
    FRONT_END_THROW("Tensor has expired.");
}

void OutPortPlace::set_target_tensor(const std::weak_ptr<TensorPlace>& target_tensor) {
    m_target_tensor = target_tensor;
}

std::vector<ov::frontend::Place::Ptr> OutPortPlace::get_consuming_ports() const {
    if (auto tensor_ptr = m_target_tensor.lock()) {
        return tensor_ptr->get_consuming_ports();
    }
    FRONT_END_THROW("Tensor has expired.");
}

bool OutPortPlace::is_equal_data(const ov::frontend::Place::Ptr& another) const {
    return get_target_tensor()->is_equal_data(another);
}

ov::frontend::Place::Ptr OutPortPlace::get_target_tensor() const {
    if (const auto& target_tensor = m_target_tensor.lock()) {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

ov::frontend::Place::Ptr OutPortPlace::get_producing_operation() const {
    if (auto op = m_op.lock()) {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}

}  // namespace onnx
}  // namespace frontend
}  // namespace ov
