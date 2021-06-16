// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <paddlepaddle_frontend/place.hpp>
#include "decoder.hpp"
#include "framework.pb.h"

using namespace ngraph;
using namespace frontend;

bool PlacePDPD::is_input() const
{
    const auto& model_ins = m_input_model.get_inputs();

    const auto cmp = [this](const Place::Ptr& p) { return p.get() == this; };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool PlacePDPD::is_output() const
{
    const auto& model_outs = m_input_model.get_outputs();
    const auto cmp = [this](const Place::Ptr& p) { return p.get() == this; };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}

OpPlacePDPD::OpPlacePDPD(const InputModel& input_model,
                         const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc,
                         const std::vector<std::string>& names)
    : PlacePDPD(input_model, names)
    , m_op_desc(op_desc)
{
}

OpPlacePDPD::OpPlacePDPD(const InputModel& input_model,
                         const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc)
    : OpPlacePDPD(input_model, op_desc, {})
{
}

const std::map<std::string, std::vector<std::shared_ptr<OutPortPlacePDPD>>>&
    OpPlacePDPD::get_output_ports() const
{
    return m_output_ports;
}

const std::map<std::string, std::vector<std::shared_ptr<InPortPlacePDPD>>>&
    OpPlacePDPD::get_input_ports() const
{
    return m_input_ports;
}

std::shared_ptr<OutPortPlacePDPD> OpPlacePDPD::get_output_port_pdpd(const std::string& name,
                                                                    int idx) const
{
    return m_output_ports.at(name)[idx];
}

std::shared_ptr<InPortPlacePDPD> OpPlacePDPD::get_input_port_pdpd(const std::string& name,
                                                                  int idx) const
{
    return m_input_ports.at(name)[idx];
}

const std::shared_ptr<paddle::framework::proto::OpDesc>& OpPlacePDPD::get_desc() const
{
    return m_op_desc;
}

void OpPlacePDPD::add_out_port(const std::shared_ptr<OutPortPlacePDPD>& output,
                               const std::string& name)
{
    m_output_ports[name].push_back(output);
}

void OpPlacePDPD::add_in_port(const std::shared_ptr<InPortPlacePDPD>& input,
                              const std::string& name)
{
    m_input_ports[name].push_back(input);
}

Place::Ptr OpPlacePDPD::get_output_port(const std::string& name) const
{
    FRONT_END_GENERAL_CHECK(
        m_output_ports.at(name).size() == 1,
        "There are multiple OutputPort Places with the same name, please specify index.");
    return m_output_ports.at(name)[0];
}

Place::Ptr OpPlacePDPD::get_input_port(const std::string& name) const
{
    FRONT_END_GENERAL_CHECK(
        m_input_ports.at(name).size() == 1,
        "There are multiple InputPort Places with the same name, please specify index.");
    return m_input_ports.at(name)[0];
}

Place::Ptr OpPlacePDPD::get_output_port() const
{
    FRONT_END_GENERAL_CHECK(m_output_ports.size() == 1 &&
                                m_output_ports.begin()->second.size() == 1,
                            "There are multiple OutputPort Places, please specify index.");
    return m_output_ports.begin()->second[0];
}

Place::Ptr OpPlacePDPD::get_input_port() const
{
    FRONT_END_GENERAL_CHECK(m_input_ports.size() == 1 && m_input_ports.begin()->second.size() == 1,
                            "There are multiple InputPort Places, please specify name and index.");
    return m_input_ports.begin()->second[0];
}

std::vector<Place::Ptr> OpPlacePDPD::get_consuming_operations() const
{
    std::vector<Place::Ptr> consuming_ops;
    for (const auto& out_port : m_output_ports)
    {
        for (const auto& out_port_place : out_port.second)
        {
            auto consuming_ops_out = out_port_place->get_consuming_operations();
            consuming_ops.insert(
                consuming_ops.end(), consuming_ops_out.begin(), consuming_ops_out.end());
        }
    }
    return consuming_ops;
}

std::vector<Place::Ptr> OpPlacePDPD::get_consuming_operations(const std::string& outputPortName,
                                                              int outputPortIndex) const
{
    return get_output_port(outputPortName, outputPortIndex)->get_consuming_operations();
}

std::vector<Place::Ptr>
    OpPlacePDPD::get_consuming_operations(const std::string& outputPortName) const
{
    return get_output_port(outputPortName)->get_consuming_operations();
}

std::vector<Place::Ptr> OpPlacePDPD::get_consuming_ports() const
{
    std::vector<Place::Ptr> consuming_ports;
    for (const auto& out_port : m_output_ports)
    {
        for (const auto& out_port_place : out_port.second)
        {
            auto consuming_ops_out = out_port_place->get_consuming_ports();
            consuming_ports.insert(
                consuming_ports.end(), consuming_ops_out.begin(), consuming_ops_out.end());
        }
    }
    return consuming_ports;
}

Place::Ptr OpPlacePDPD::get_output_port(const std::string& outputName, int outputPortIndex) const
{
    return m_output_ports.at(outputName)[outputPortIndex];
}

Place::Ptr OpPlacePDPD::get_input_port(const std::string& inputName, int inputPortIndex) const
{
    return m_input_ports.at(inputName)[inputPortIndex];
}

Place::Ptr OpPlacePDPD::get_source_tensor() const
{
    return get_input_port()->get_source_tensor();
}

Place::Ptr OpPlacePDPD::get_source_tensor(const std::string& inputName) const
{
    return get_input_port(inputName)->get_source_tensor();
}

Place::Ptr OpPlacePDPD::get_source_tensor(const std::string& inputName, int inputPortIndex) const
{
    return get_input_port(inputName, inputPortIndex)->get_source_tensor();
}

Place::Ptr OpPlacePDPD::get_target_tensor() const
{
    return get_output_port()->get_target_tensor();
}

Place::Ptr OpPlacePDPD::get_target_tensor(const std::string& outputName) const
{
    return get_output_port(outputName)->get_target_tensor();
}

Place::Ptr OpPlacePDPD::get_target_tensor(const std::string& outputName, int outputPortIndex) const
{
    return get_output_port(outputName, outputPortIndex)->get_target_tensor();
}

Place::Ptr OpPlacePDPD::get_producing_operation(const std::string& inputName) const
{
    return get_input_port(inputName)->get_producing_operation();
}

Place::Ptr OpPlacePDPD::get_producing_operation(const std::string& inputName,
                                                int inputPortIndex) const
{
    return get_input_port(inputName, inputPortIndex)->get_producing_operation();
}

Place::Ptr OpPlacePDPD::get_producing_operation() const
{
    return get_input_port()->get_producing_operation();
}

TensorPlacePDPD::TensorPlacePDPD(const InputModel& input_model,
                                 const std::vector<std::string>& names,
                                 const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc)
    : PlacePDPD(input_model, names)
    , m_var_desc(var_desc)
{
    const auto& var_type = var_desc->type();
    if (var_type.type() == paddle::framework::proto::VarType::LOD_TENSOR)
    {
        const auto& tensor_desc = var_type.lod_tensor().tensor();
        m_type = TYPE_MAP[tensor_desc.data_type()];
        m_pshape = PartialShape(
            std::vector<Dimension>(tensor_desc.dims().begin(), tensor_desc.dims().end()));
    }
}

TensorPlacePDPD::TensorPlacePDPD(const InputModel& input_model,
                                 const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc)
    : TensorPlacePDPD(input_model, {var_desc->name()}, var_desc)
{
}

std::vector<Place::Ptr> TensorPlacePDPD::get_consuming_ports() const
{
    std::vector<Place::Ptr> consuming_ports;
    for (const auto& consuming_port : m_consuming_ports)
    {
        if (const auto& locked = consuming_port.lock())
        {
            consuming_ports.push_back(locked);
        }
        else
        {
            FRONT_END_THROW("Consuming Port has expired.");
        }
    }
    return consuming_ports;
}

Place::Ptr TensorPlacePDPD::get_producing_port() const
{
    FRONT_END_GENERAL_CHECK(m_producing_ports.size() == 1, "Only one producing port is supported.");
    if (const auto& producing_port = m_producing_ports[0].lock())
    {
        return producing_port;
    }
    FRONT_END_THROW("Producing Port has expired.");
}

void TensorPlacePDPD::add_producing_port(const std::shared_ptr<OutPortPlacePDPD>& out_port)
{
    m_producing_ports.push_back(out_port);
}

void TensorPlacePDPD::add_consuming_port(const std::shared_ptr<InPortPlacePDPD>& in_port)
{
    m_consuming_ports.push_back(in_port);
}

const std::shared_ptr<paddle::framework::proto::VarDesc>& TensorPlacePDPD::get_desc() const
{
    return m_var_desc;
}

std::vector<Place::Ptr> TensorPlacePDPD::get_consuming_operations() const
{
    std::vector<Place::Ptr> consuming_ops;
    for (const auto& consuming_port : m_consuming_ports)
    {
        if (auto port_ptr = consuming_port.lock())
        {
            auto port_consuming_ops = port_ptr->get_consuming_operations();
            consuming_ops.insert(
                consuming_ops.end(), port_consuming_ops.begin(), port_consuming_ops.end());
        }
        else
        {
            FRONT_END_THROW("Port has expired.");
        }
    }
    return consuming_ops;
}

bool TensorPlacePDPD::is_equal_data(Place::Ptr another) const
{
    auto consuming_ports = get_consuming_ports();
    bool eq_to_consuming_port =
        std::any_of(consuming_ports.begin(), consuming_ports.end(), [&another](const Ptr& place) {
            return place->is_equal(another);
        });
    return is_equal(another) || get_producing_port()->is_equal(another) || eq_to_consuming_port;
}

Place::Ptr TensorPlacePDPD::get_producing_operation() const
{
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<TensorPlacePDPD> InPortPlacePDPD::get_source_tensor_pdpd() const
{
    if (const auto& tensor = m_source_tensor.lock())
    {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

std::shared_ptr<OpPlacePDPD> InPortPlacePDPD::get_op()
{
    if (const auto& op = m_op.lock())
    {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}

void InPortPlacePDPD::set_source_tensor(const std::weak_ptr<TensorPlacePDPD>& source_tensor)
{
    m_source_tensor = source_tensor;
}

std::vector<Place::Ptr> InPortPlacePDPD::get_consuming_operations() const
{
    if (const auto& consuming_op = m_op.lock())
    {
        return {consuming_op};
    }
    FRONT_END_THROW("Operation has expired.");
}

Place::Ptr InPortPlacePDPD::get_source_tensor() const
{
    if (const auto& tensor = m_source_tensor.lock())
    {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

Place::Ptr InPortPlacePDPD::get_producing_port() const
{
    return get_source_tensor()->get_producing_port();
}

bool InPortPlacePDPD::is_equal_data(Place::Ptr another) const
{
    return get_source_tensor()->is_equal_data(another);
}

Place::Ptr InPortPlacePDPD::get_producing_operation() const
{
    return get_producing_port()->get_producing_operation();
}

std::shared_ptr<TensorPlacePDPD> OutPortPlacePDPD::get_target_tensor_pdpd() const
{
    if (const auto& target_tensor = m_target_tensor.lock())
    {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

std::vector<Place::Ptr> OutPortPlacePDPD::get_consuming_operations() const
{
    if (auto tensor_ptr = m_target_tensor.lock())
    {
        return tensor_ptr->get_consuming_operations();
    }
    FRONT_END_THROW("Tensor has expired.");
}

void OutPortPlacePDPD::set_target_tensor(const std::weak_ptr<TensorPlacePDPD>& target_tensor)
{
    m_target_tensor = target_tensor;
}

std::vector<Place::Ptr> OutPortPlacePDPD::get_consuming_ports() const
{
    if (auto tensor_ptr = m_target_tensor.lock())
    {
        return tensor_ptr->get_consuming_ports();
    }
    FRONT_END_THROW("Tensor has expired.");
}

bool OutPortPlacePDPD::is_equal_data(Place::Ptr another) const
{
    return get_target_tensor()->is_equal_data(another);
}

Place::Ptr OutPortPlacePDPD::get_target_tensor() const
{
    if (const auto& target_tensor = m_target_tensor.lock())
    {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}

Place::Ptr OutPortPlacePDPD::get_producing_operation() const
{
    if (auto op = m_op.lock())
    {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}
