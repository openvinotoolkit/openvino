// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/util/env_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/frontend/place.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

//----------- InputModel ---------------------------
std::vector<Place::Ptr> InputModel::get_inputs() const {
    return m_actual->get_inputs();
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    return m_actual->get_outputs();
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    return m_actual->get_place_by_tensor_name(tensor_name);
}

Place::Ptr InputModel::get_place_by_operation_name(const std::string& operation_name) const {
    return m_actual->get_place_by_operation_name(operation_name);
}

Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                  int input_port_index) {
    return m_actual->get_place_by_operation_name_and_input_port(operation_name, input_port_index);
}

Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                   int output_port_index) {
    return m_actual->get_place_by_operation_name_and_output_port(operation_name, output_port_index);
}

void InputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    m_actual->set_name_for_tensor(tensor, new_name);
}

void InputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    m_actual->add_name_for_tensor(tensor, new_name);
}

void InputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    m_actual->set_name_for_operation(operation, new_name);
}

void InputModel::free_name_for_tensor(const std::string& name) {
    m_actual->free_name_for_tensor(name);
}

void InputModel::free_name_for_operation(const std::string& name) {
    m_actual->free_name_for_operation(name);
}

void InputModel::set_name_for_dimension(const Place::Ptr& place, size_t shape_dim_index, const std::string& dim_name) {
    m_actual->set_name_for_dimension(place, shape_dim_index, dim_name);
}

void InputModel::cut_and_add_new_input(const Place::Ptr& place, const std::string& new_name_optional) {
    m_actual->cut_and_add_new_input(place, new_name_optional);
}

void InputModel::cut_and_add_new_output(const Place::Ptr& place, const std::string& new_name_optional) {
    m_actual->cut_and_add_new_output(place, new_name_optional);
}

Place::Ptr InputModel::add_output(const Place::Ptr& place) {
    return m_actual->add_output(place);
}

void InputModel::remove_output(const Place::Ptr& place) {
    m_actual->remove_output(place);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    m_actual->override_all_outputs(outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    m_actual->override_all_inputs(inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    m_actual->extract_subgraph(inputs, outputs);
}

// Setting tensor properties
void InputModel::set_partial_shape(const Place::Ptr& place, const PartialShape& shape) {
    m_actual->set_partial_shape(place, shape);
}

PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    return m_actual->get_partial_shape(place);
}

void InputModel::set_element_type(const Place::Ptr& place, const element::Type& type) {
    m_actual->set_element_type(place, type);
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    m_actual->set_tensor_value(place, value);
}

void InputModel::set_tensor_partial_value(const Place::Ptr& place, const void* min_value, const void* max_value) {
    m_actual->set_tensor_partial_value(place, min_value, max_value);
}

//----------- IInputModel ---------------------------
std::vector<Place::Ptr> IInputModel::get_inputs() const {
    return {};
}

std::vector<Place::Ptr> IInputModel::get_outputs() const {
    return {};
}

Place::Ptr IInputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    return nullptr;
}

Place::Ptr IInputModel::get_place_by_operation_name(const std::string& operation_name) const {
    return nullptr;
}

Place::Ptr IInputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                   int input_port_index) {
    return nullptr;
}

Place::Ptr IInputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                    int output_port_index) {
    return nullptr;
}

void IInputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_tensor);
}

void IInputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(add_name_for_tensor);
}

void IInputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_operation);
}

void IInputModel::free_name_for_tensor(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_tensor);
}

void IInputModel::free_name_for_operation(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_operation);
}

void IInputModel::set_name_for_dimension(const Place::Ptr& place, size_t shape_dim_index, const std::string& dim_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_dimension);
}

void IInputModel::cut_and_add_new_input(const Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_input);
}

void IInputModel::cut_and_add_new_output(const Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_output);
}

Place::Ptr IInputModel::add_output(const Place::Ptr& place) {
    FRONT_END_NOT_IMPLEMENTED(add_output);
}

void IInputModel::remove_output(const Place::Ptr& place) {
    FRONT_END_NOT_IMPLEMENTED(remove_output);
}

void IInputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_outputs);
}

void IInputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_inputs);
}

void IInputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(extract_subgraph);
}

// Setting tensor properties
void IInputModel::set_partial_shape(const Place::Ptr& place, const PartialShape&) {
    FRONT_END_NOT_IMPLEMENTED(set_partial_shape);
}

PartialShape IInputModel::get_partial_shape(const Place::Ptr& place) const {
    FRONT_END_NOT_IMPLEMENTED(get_partial_shape);
}

void IInputModel::set_element_type(const Place::Ptr& place, const element::Type&) {
    FRONT_END_NOT_IMPLEMENTED(set_element_type);
}

void IInputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_value);
}

void IInputModel::set_tensor_partial_value(const Place::Ptr& place, const void* min_value, const void* max_value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_partial_value);
}
