// Copyright (C) 2018-2022 Intel Corporation
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

// Generic place implementation.
GenericPlace::GenericPlace(Place::Ptr impl, std::shared_ptr<void> shared_object)
    : m_impl{std::move(impl)},
      m_shared_object{std::move(shared_object)} {
    FRONT_END_GENERAL_CHECK(m_impl, "Implementation is nullptr!");
}

std::vector<std::string> GenericPlace::get_names() const {
    return m_impl->get_names();
}

std::vector<Place::Ptr> GenericPlace::get_consuming_operations() const {
    // ???? impl return vector of pointers which has to be again wrapped in smart pointers with library handle.
    // How do it in efficient way?
    return m_impl->get_consuming_operations();
}

std::vector<Place::Ptr> GenericPlace::get_consuming_operations(int output_port_index) const {
    return m_impl->get_consuming_operations(output_port_index);
}

std::vector<Place::Ptr> GenericPlace::get_consuming_operations(const std::string& outputName) const {
    return m_impl->get_consuming_operations(outputName);
}

std::vector<Place::Ptr> GenericPlace::get_consuming_operations(const std::string& outputName,
                                                               int outputPortIndex) const {
    return m_impl->get_consuming_operations(outputName, outputPortIndex);
}

Place::Ptr GenericPlace::get_target_tensor() const {
    return std::make_shared<GenericPlace>(m_impl->get_target_tensor(), m_shared_object);
}

Place::Ptr GenericPlace::get_target_tensor(const std::string& outputName) const {
    return m_impl->get_target_tensor(outputName);
}

Place::Ptr GenericPlace::get_target_tensor(const std::string& outputName, int outputPortIndex) const {
    return m_impl->get_target_tensor(outputName, outputPortIndex);
}

Place::Ptr GenericPlace::get_target_tensor(int output_port_index) const {
    return m_impl->get_target_tensor(output_port_index);
}

Place::Ptr GenericPlace::get_source_tensor() const {
    return m_impl->get_source_tensor();
}

Place::Ptr GenericPlace::get_source_tensor(int input_port_index) const {
    return m_impl->get_source_tensor(input_port_index);
}

Place::Ptr GenericPlace::get_source_tensor(const std::string& inputName) const {
    return m_impl->get_source_tensor(inputName);
}

Place::Ptr GenericPlace::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    return m_impl->get_source_tensor(inputName, inputPortIndex);
}

Place::Ptr GenericPlace::get_producing_operation() const {
    return m_impl->get_producing_operation();
}

Place::Ptr GenericPlace::get_producing_operation(int input_port_index) const {
    return m_impl->get_producing_operation(input_port_index);
}

Place::Ptr GenericPlace::get_producing_operation(const std::string& inputName) const {
    return m_impl->get_producing_operation(inputName);
}

Place::Ptr GenericPlace::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    return m_impl->get_producing_operation(inputName, inputPortIndex);
}

Place::Ptr GenericPlace::get_producing_port() const {
    // Place from common part must create shared pointer with library dependency to provide safe destruction of
    // object pointed by smart pointer and smart pointer itself. When smart pointer returned directly from impl
    // then there will be seg fault on smart pointer destruction.
    // Must be applied to all methods which return smart pointer.
    auto impl = m_impl->get_producing_port();
    if (impl) {
        return std::make_shared<GenericPlace>(impl, m_shared_object);
    }
    return {};
}

Place::Ptr GenericPlace::get_input_port() const {
    return m_impl->get_input_port();
}

Place::Ptr GenericPlace::get_input_port(int input_port_index) const {
    return m_impl->get_input_port(input_port_index);
}

Place::Ptr GenericPlace::get_input_port(const std::string& input_name) const {
    return m_impl->get_input_port(input_name);
}

Place::Ptr GenericPlace::get_input_port(const std::string& input_name, int input_port_index) const {
    return m_impl->get_input_port(input_name, input_port_index);
}

Place::Ptr GenericPlace::get_output_port() const {
    return m_impl->get_output_port();
}

Place::Ptr GenericPlace::get_output_port(int output_port_index) const {
    return m_impl->get_output_port(output_port_index);
}

Place::Ptr GenericPlace::get_output_port(const std::string& output_name) const {
    return m_impl->get_output_port(output_name);
}

Place::Ptr GenericPlace::get_output_port(const std::string& output_name, int output_port_index) const {
    return m_impl->get_output_port(output_name, output_port_index);
}

std::vector<Place::Ptr> GenericPlace::get_consuming_ports() const {
    return m_impl->get_consuming_ports();
}

bool GenericPlace::is_input() const {
    return m_impl->is_input();
}

bool GenericPlace::is_output() const {
    return m_impl->is_output();
}

bool GenericPlace::is_equal(const Ptr& another) const {
    return m_impl->is_equal(another);
}

bool GenericPlace::is_equal_data(const Ptr& another) const {
    return m_impl->is_equal_data(another);
}
