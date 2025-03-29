// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/place.hpp"

#include <openvino/util/env_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

Place::~Place() = default;

std::vector<std::string> Place::get_names() const {
    FRONT_END_NOT_IMPLEMENTED(get_names);
}

std::vector<Place::Ptr> Place::get_consuming_operations() const {
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(int output_port_index) const {
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName, int outputPortIndex) const {
    return {};
}

Place::Ptr Place::get_target_tensor() const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(int output_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation() const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_producing_port() const {
    return nullptr;
}

Place::Ptr Place::get_input_port() const {
    return nullptr;
}

Place::Ptr Place::get_input_port(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name) const {
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name, int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_output_port() const {
    return nullptr;
}

Place::Ptr Place::get_output_port(int output_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name) const {
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name, int output_port_index) const {
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_ports() const {
    return {};
}

bool Place::is_input() const {
    FRONT_END_NOT_IMPLEMENTED(is_input);
}

bool Place::is_output() const {
    FRONT_END_NOT_IMPLEMENTED(is_output);
}

bool Place::is_equal(const Ptr& another) const {
    FRONT_END_NOT_IMPLEMENTED(is_equal);
}

bool Place::is_equal_data(const Ptr& another) const {
    FRONT_END_NOT_IMPLEMENTED(is_equal_data);
}

Place::Ptr Place::get_source_tensor() const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(int input_port_index) const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName) const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName) const {
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName, int outputPortIndex) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName) const {
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName) const {
    return {};
}
