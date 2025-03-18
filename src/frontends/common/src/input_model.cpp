// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/util/env_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/place.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

InputModel::~InputModel() = default;

std::vector<Place::Ptr> InputModel::get_inputs() const {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_inputs", m_actual->get_inputs())
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_outputs", m_actual->get_outputs())
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_place_by_tensor_name", m_actual->get_place_by_tensor_name(tensor_name))
}

Place::Ptr InputModel::get_place_by_input_index(size_t input_idx) const {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_place_by_input_index", m_actual->get_place_by_input_index(input_idx))
}

Place::Ptr InputModel::get_place_by_operation_name(const std::string& operation_name) const {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_place_by_operation_name", m_actual->get_place_by_operation_name(operation_name))
}

Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                  int input_port_index) {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_place_by_operation_name_and_input_port",
                              m_actual->get_place_by_operation_name_and_input_port(operation_name, input_port_index))
}

Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                   int output_port_index) {
    if (!m_actual) {
        return {};
    }
    FRONTEND_RETURN_STATEMENT("get_place_by_operation_name_and_output_port",
                              m_actual->get_place_by_operation_name_and_output_port(operation_name, output_port_index))
}

void InputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_name_for_tensor);
    FRONTEND_CALL_STATEMENT("set_name_for_tensor", m_actual->set_name_for_tensor(tensor, new_name))
}

void InputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, add_name_for_tensor);
    FRONTEND_CALL_STATEMENT("add_name_for_tensor", m_actual->add_name_for_tensor(tensor, new_name))
}

void InputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_name_for_operation);
    FRONTEND_CALL_STATEMENT("set_name_for_operation", m_actual->set_name_for_operation(operation, new_name))
}

void InputModel::free_name_for_tensor(const std::string& name) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, free_name_for_tensor);
    FRONTEND_CALL_STATEMENT("free_name_for_tensor", m_actual->free_name_for_tensor(name))
}

void InputModel::free_name_for_operation(const std::string& name) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, free_name_for_operation);
    FRONTEND_CALL_STATEMENT("free_name_for_operation", m_actual->free_name_for_operation(name))
}

void InputModel::set_name_for_dimension(const Place::Ptr& place, size_t shape_dim_index, const std::string& dim_name) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_name_for_dimension);
    FRONTEND_CALL_STATEMENT("set_name_for_dimension",
                            m_actual->set_name_for_dimension(place, shape_dim_index, dim_name))
}

void InputModel::cut_and_add_new_input(const Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, cut_and_add_new_input);
    FRONTEND_CALL_STATEMENT("cut_and_add_new_input", m_actual->cut_and_add_new_input(place, new_name_optional))
}

void InputModel::cut_and_add_new_output(const Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, cut_and_add_new_output);
    FRONTEND_CALL_STATEMENT("cut_and_add_new_output", m_actual->cut_and_add_new_output(place, new_name_optional))
}

Place::Ptr InputModel::add_output(const Place::Ptr& place) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, add_output);
    FRONTEND_RETURN_STATEMENT("add_output", m_actual->add_output(place))
}

void InputModel::remove_output(const Place::Ptr& place) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, remove_output);
    FRONTEND_CALL_STATEMENT("remove_output", m_actual->remove_output(place))
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, override_all_outputs);
    FRONTEND_CALL_STATEMENT("override_all_outputs", m_actual->override_all_outputs(outputs))
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, override_all_inputs);
    FRONTEND_CALL_STATEMENT("override_all_inputs", m_actual->override_all_inputs(inputs))
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, extract_subgraph);
    FRONTEND_CALL_STATEMENT("extract_subgraph", m_actual->extract_subgraph(inputs, outputs))
}

// Setting tensor properties
void InputModel::set_partial_shape(const Place::Ptr& place, const PartialShape& shape) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_partial_shape);
    FRONTEND_CALL_STATEMENT("set_partial_shape", m_actual->set_partial_shape(place, shape))
}

PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, get_partial_shape);
    FRONTEND_RETURN_STATEMENT("get_partial_shape", m_actual->get_partial_shape(place))
}

void InputModel::set_element_type(const Place::Ptr& place, const element::Type& type) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_element_type);
    FRONTEND_CALL_STATEMENT("set_element_type", m_actual->set_element_type(place, type))
}

element::Type InputModel::get_element_type(const Place::Ptr& place) const {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, get_element_type);
    FRONTEND_RETURN_STATEMENT("get_element_type", m_actual->get_element_type(place))
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_tensor_value);
    FRONTEND_CALL_STATEMENT("set_tensor_value", m_actual->set_tensor_value(place, value))
}

void InputModel::set_tensor_partial_value(const Place::Ptr& place, const void* min_value, const void* max_value) {
    FRONT_END_CHECK_IMPLEMENTED(m_actual, set_tensor_partial_value);
    FRONTEND_CALL_STATEMENT("set_tensor_partial_value", m_actual->set_tensor_partial_value(place, min_value, max_value))
}
