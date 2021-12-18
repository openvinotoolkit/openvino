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

InputModel::~InputModel() = default;

std::vector<Place::Ptr> InputModel::get_inputs() const {
    return {};
}

std::vector<Place::Ptr> InputModel::get_outputs() const {
    return {};
}

Place::Ptr InputModel::get_place_by_tensor_name(const std::string& tensor_name) const {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name(const std::string& operation_name) const {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name_and_input_port(const std::string& operation_name,
                                                                  int input_port_index) {
    return nullptr;
}

Place::Ptr InputModel::get_place_by_operation_name_and_output_port(const std::string& operation_name,
                                                                   int output_port_index) {
    return nullptr;
}

void InputModel::set_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_tensor);
}

void InputModel::add_name_for_tensor(const Place::Ptr& tensor, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(add_name_for_tensor);
}

void InputModel::set_name_for_operation(const Place::Ptr& operation, const std::string& new_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_operation);
}

void InputModel::free_name_for_tensor(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_tensor);
}

void InputModel::free_name_for_operation(const std::string& name) {
    FRONT_END_NOT_IMPLEMENTED(free_name_for_operation);
}

void InputModel::set_name_for_dimension(const Place::Ptr& place, size_t shape_dim_index, const std::string& dim_name) {
    FRONT_END_NOT_IMPLEMENTED(set_name_for_dimension);
}

void InputModel::cut_and_add_new_input(const Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_input);
}

void InputModel::cut_and_add_new_output(const Place::Ptr& place, const std::string& new_name_optional) {
    FRONT_END_NOT_IMPLEMENTED(cut_and_add_new_output);
}

Place::Ptr InputModel::add_output(const Place::Ptr& place) {
    FRONT_END_NOT_IMPLEMENTED(add_output);
}

void InputModel::remove_output(const Place::Ptr& place) {
    FRONT_END_NOT_IMPLEMENTED(remove_output);
}

void InputModel::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_outputs);
}

void InputModel::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    FRONT_END_NOT_IMPLEMENTED(override_all_inputs);
}

void InputModel::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    FRONT_END_NOT_IMPLEMENTED(extract_subgraph);
}

// Setting tensor properties
void InputModel::set_partial_shape(const Place::Ptr& place, const PartialShape&) {
    FRONT_END_NOT_IMPLEMENTED(set_partial_shape);
}

PartialShape InputModel::get_partial_shape(const Place::Ptr& place) const {
    FRONT_END_NOT_IMPLEMENTED(get_partial_shape);
}

void InputModel::set_element_type(const Place::Ptr& place, const element::Type&) {
    FRONT_END_NOT_IMPLEMENTED(set_element_type);
}

void InputModel::set_tensor_value(const Place::Ptr& place, const void* value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_value);
}

void InputModel::set_tensor_partial_value(const Place::Ptr& place, const void* min_value, const void* max_value) {
    FRONT_END_NOT_IMPLEMENTED(set_tensor_partial_value);
}
