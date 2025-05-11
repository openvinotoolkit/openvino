// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "place.hpp"

#include "decoder_proto.hpp"
#include "framework.pb.h"

namespace ov {
namespace frontend {
namespace paddle {

bool Place::is_input() const {
    const auto& model_ins = m_input_model.get_inputs();

    const auto cmp = [this](const Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool Place::is_output() const {
    const auto& model_outs = m_input_model.get_outputs();
    const auto cmp = [this](const Place::Ptr& p) {
        return p.get() == this;
    };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}

OpPlace::OpPlace(const ov::frontend::InputModel& input_model,
                 const ::paddle::framework::proto::OpDesc& op_desc,
                 const std::vector<std::string>& names)
    : Place(input_model, names),
      m_op_desc(op_desc) {}

OpPlace::OpPlace(const ov::frontend::InputModel& input_model, const ::paddle::framework::proto::OpDesc& op_desc)
    : OpPlace(input_model, op_desc, {}) {}

const std::map<std::string, std::vector<std::shared_ptr<OutPortPlace>>>& OpPlace::get_output_ports() const {
    return m_output_ports;
}

const std::map<std::string, std::vector<std::shared_ptr<InPortPlace>>>& OpPlace::get_input_ports() const {
    return m_input_ports;
}

std::shared_ptr<OutPortPlace> OpPlace::get_output_port_paddle(const std::string& outputName,
                                                              int outputPortIndex) const {
    FRONT_END_GENERAL_CHECK((size_t)outputPortIndex <= m_output_ports.at(outputName).size(),
                            "outputPortIndex is out of bounds.");
    return m_output_ports.at(outputName)[outputPortIndex];
}

std::shared_ptr<InPortPlace> OpPlace::get_input_port_paddle(const std::string& inputName, int inputPortIndex) const {
    FRONT_END_GENERAL_CHECK((size_t)inputPortIndex <= m_input_ports.at(inputName).size(),
                            "inputPortIndex is out of bounds.");
    return m_input_ports.at(inputName)[inputPortIndex];
}

const ::paddle::framework::proto::OpDesc& OpPlace::get_desc() const {
    return m_op_desc;
}

const std::shared_ptr<DecoderBase> OpPlace::get_decoder() const {
    return m_op_decoder;
}

void OpPlace::set_decoder(const std::shared_ptr<DecoderBase> op_decoder) {
    m_op_decoder = op_decoder;
}

void OpPlace::add_out_port(const std::shared_ptr<OutPortPlace>& output, const std::string& name) {
    m_output_ports[name].push_back(output);
}

void OpPlace::add_in_port(const std::shared_ptr<InPortPlace>& input, const std::string& name) {
    m_input_ports[name].push_back(input);
}

Place::Ptr OpPlace::get_output_port(const std::string& name) const {
    FRONT_END_GENERAL_CHECK(m_output_ports.at(name).size() == 1, "Only one output port should exist.");
    return m_output_ports.at(name)[0];
}

Place::Ptr OpPlace::get_input_port(const std::string& name) const {
    FRONT_END_GENERAL_CHECK(m_input_ports.at(name).size() == 1, "Only one input port should exist.");
    return m_input_ports.at(name)[0];
}

Place::Ptr OpPlace::get_input_port(int outputPortIndex) const {
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1, "Only one named input port should exist.");
    return m_input_ports.begin()->second[outputPortIndex];
}

Place::Ptr OpPlace::get_output_port(int outputPortIndex) const {
    FRONT_END_GENERAL_CHECK(m_output_ports.size() == 1, "Only one named output port should exist.");
    return m_output_ports.begin()->second[outputPortIndex];
}

Place::Ptr OpPlace::get_output_port() const {
    FRONT_END_GENERAL_CHECK(m_output_ports.size() == 1 && m_output_ports.begin()->second.size() == 1,
                            "Only one output port should exist.");
    return m_output_ports.begin()->second[0];
}

Place::Ptr OpPlace::get_input_port() const {
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1 && m_input_ports.begin()->second.size() == 1,
                            "Only one input port should exist.");
    return m_input_ports.begin()->second[0];
}

std::vector<Place::Ptr> OpPlace::get_consuming_operations() const {
    std::vector<Place::Ptr> consuming_ops;
    for (const auto& out_port : m_output_ports) {
        for (const auto& out_port_place : out_port.second) {
            auto consuming_ops_out = out_port_place->get_consuming_operations();
            consuming_ops.insert(consuming_ops.end(), consuming_ops_out.begin(), consuming_ops_out.end());
        }
    }
    return consuming_ops;
}

std::vector<Place::Ptr> OpPlace::get_consuming_operations(const std::string& outputPortName,
                                                          int outputPortIndex) const {
    return get_output_port(outputPortName, outputPortIndex)->get_consuming_operations();
}

std::vector<Place::Ptr> OpPlace::get_consuming_operations(int outputPortIndex) const {
    return get_output_port(outputPortIndex)->get_consuming_operations();
}

std::vector<Place::Ptr> OpPlace::get_consuming_operations(const std::string& outputPortName) const {
    return get_output_port(outputPortName)->get_consuming_operations();
}

std::vector<Place::Ptr> OpPlace::get_consuming_ports() const {
    std::vector<Place::Ptr> consuming_ports;
    for (const auto& out_port : m_output_ports) {
        for (const auto& out_port_place : out_port.second) {
            auto consuming_ops_out = out_port_place->get_consuming_ports();
            consuming_ports.insert(consuming_ports.end(), consuming_ops_out.begin(), consuming_ops_out.end());
        }
    }
    return consuming_ports;
}

Place::Ptr OpPlace::get_output_port(const std::string& outputName, int outputPortIndex) const {
    FRONT_END_GENERAL_CHECK((size_t)outputPortIndex <= m_output_ports.at(outputName).size(),
                            "outputPortIndex is Out of bounds.");
    return m_output_ports.at(outputName)[outputPortIndex];
}

Place::Ptr OpPlace::get_input_port(const std::string& inputName, int inputPortIndex) const {
    FRONT_END_GENERAL_CHECK((size_t)inputPortIndex <= m_input_ports.at(inputName).size(),
                            "inputPortIndex is out of bounds.");
    return m_input_ports.at(inputName)[inputPortIndex];
}

Place::Ptr OpPlace::get_source_tensor() const {
    return get_input_port()->get_source_tensor();
}

Place::Ptr OpPlace::get_source_tensor(const std::string& inputName) const {
    return get_input_port(inputName)->get_source_tensor();
}

Place::Ptr OpPlace::get_source_tensor(int inputPortIndex) const {
    return get_input_port(inputPortIndex)->get_source_tensor();
}

Place::Ptr OpPlace::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return get_input_port(inputName, inputPortIndex)->get_source_tensor();
}

Place::Ptr OpPlace::get_target_tensor() const {
    return get_output_port()->get_target_tensor();
}

Place::Ptr OpPlace::get_target_tensor(const std::string& outputName) const {
    return get_output_port(outputName)->get_target_tensor();
}

Place::Ptr OpPlace::get_target_tensor(const std::string& outputName, int outputPortIndex) const {
    return get_output_port(outputName, outputPortIndex)->get_target_tensor();
}

Place::Ptr OpPlace::get_producing_operation(const std::string& inputName) const {
    return get_input_port(inputName)->get_producing_operation();
}

Place::Ptr OpPlace::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    return get_input_port(inputName, inputPortIndex)->get_producing_operation();
}

Place::Ptr OpPlace::get_producing_operation() const {
    return get_input_port()->get_producing_operation();
}

Place::Ptr OpPlace::get_producing_operation(int inputPortIndex) const {
    return get_input_port(inputPortIndex)->get_producing_operation();
}

Place::Ptr OpPlace::get_target_tensor(int outputPortIndex) const {
    return get_output_port(outputPortIndex)->get_target_tensor();
}

TensorPlace::TensorPlace(const ov::frontend::InputModel& input_model,
                         const std::vector<std::string>& names,
                         const ::paddle::framework::proto::VarDesc& var_desc)
    : Place(input_model, names),
      m_var_desc(var_desc) {
    const auto& var_type = var_desc.type();
    if (var_type.type() == ::paddle::framework::proto::VarType::LOD_TENSOR) {
        const auto& tensor_desc = var_type.lod_tensor().tensor();
        m_type = get_ov_type(tensor_desc.data_type());
        m_pshape = PartialShape(std::vector<Dimension>(tensor_desc.dims().begin(), tensor_desc.dims().end()));
    }
}

TensorPlace::TensorPlace(const ov::frontend::InputModel& input_model,
                         const ::paddle::framework::proto::VarDesc& var_desc)
    : TensorPlace(input_model, {var_desc.name()}, var_desc) {}

std::vector<Place::Ptr> TensorPlace::get_consuming_ports() const {
    std::vector<Place::Ptr> consuming_ports;
    for (const auto& consuming_port : m_consuming_ports) {
        if (const auto& locked = consuming_port.lock()) {
            consuming_ports.push_back(locked);
        } else {
            FRONT_END_THROW("Consuming Port has expired.");
        }
    }
    return consuming_ports;
}

Place::Ptr TensorPlace::get_producing_port() const {
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

const ::paddle::framework::proto::VarDesc& TensorPlace::get_desc() const {
    return m_var_desc;
}

std::vector<Place::Ptr> TensorPlace::get_consuming_operations() const {
    std::vector<Place::Ptr> consuming_ops;
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

bool TensorPlace::is_equal_data(const Place::Ptr& another) const {
    auto consuming_ports = get_consuming_ports();
    bool eq_to_consuming_port =
        std::any_of(consuming_ports.begin(), consuming_ports.end(), [&another](const Ptr& place) {
            return place->is_equal(another);
        });
    return is_equal(another) || get_producing_port()->is_equal(another) || eq_to_consuming_port;
}

Place::Ptr TensorPlace::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<TensorPlace> InPortPlace::get_source_tensor_paddle() const {
    if (const auto& tensor = m_source_tensor.lock()) {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
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

std::vector<Place::Ptr> InPortPlace::get_consuming_operations() const {
    if (const auto& consuming_op = m_op.lock()) {
        return {consuming_op};
    }
    FRONT_END_THROW("Operation has expired.");
}

Place::Ptr InPortPlace::get_source_tensor() const {
    if (const auto& tensor = m_source_tensor.lock()) {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

Place::Ptr InPortPlace::get_producing_port() const {
    return get_source_tensor()->get_producing_port();
}

bool InPortPlace::is_equal_data(const Place::Ptr& another) const {
    return get_source_tensor()->is_equal_data(another);
}

Place::Ptr InPortPlace::get_producing_operation() const {
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<TensorPlace> OutPortPlace::get_target_tensor_paddle() const {
    if (const auto& target_tensor = m_target_tensor.lock()) {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

std::vector<Place::Ptr> OutPortPlace::get_consuming_operations() const {
    if (auto tensor_ptr = m_target_tensor.lock()) {
        return tensor_ptr->get_consuming_operations();
    }
    FRONT_END_THROW("Tensor has expired.");
}

void OutPortPlace::set_target_tensor(const std::weak_ptr<TensorPlace>& target_tensor) {
    m_target_tensor = target_tensor;
}

std::vector<Place::Ptr> OutPortPlace::get_consuming_ports() const {
    if (auto tensor_ptr = m_target_tensor.lock()) {
        return tensor_ptr->get_consuming_ports();
    }
    FRONT_END_THROW("Tensor has expired.");
}

bool OutPortPlace::is_equal_data(const Place::Ptr& another) const {
    return get_target_tensor()->is_equal_data(another);
}

Place::Ptr OutPortPlace::get_target_tensor() const {
    if (const auto& target_tensor = m_target_tensor.lock()) {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

Place::Ptr OutPortPlace::get_producing_operation() const {
    if (auto op = m_op.lock()) {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov
