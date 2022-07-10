// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/place.hpp"

#include <openvino/util/env_util.hpp>

#include "openvino/frontend/exception.hpp"
#include "utils.hpp"

using namespace ov;
using namespace ov::frontend;

Place::Place(Place::Ptr pimpl, std::shared_ptr<void> shared_object)
    : m_pimpl{std::move(pimpl)},
      SharedObjectExtension(std::move(shared_object)) {}

std::vector<std::string> Place::get_names() const {
    if (m_pimpl) {
        return m_pimpl->get_names();
    }
    FRONT_END_NOT_IMPLEMENTED(get_names);
}

std::vector<Place::Ptr> Place::get_consuming_operations() const {
    if (m_pimpl) {
        return transform_pimpls(m_pimpl->get_consuming_operations());
    }
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(int output_port_index) const {
    if (m_pimpl) {
        return transform_pimpls(m_pimpl->get_consuming_operations(output_port_index));
    }
    return {};
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName, int outputPortIndex) const {
    if (m_pimpl) {
        return transform_pimpls(m_pimpl->get_consuming_operations(outputPortName, outputPortIndex));
    }
    return {};
}

Place::Ptr Place::get_target_tensor() const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_target_tensor());
    }
    return nullptr;
}

Place::Ptr Place::get_target_tensor(int output_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_target_tensor(output_port_index));
    }
    return nullptr;
}

Place::Ptr Place::get_producing_operation() const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_producing_operation());
    }
    return nullptr;
}

Place::Ptr Place::get_producing_operation(int input_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_producing_operation(input_port_index));
    }
    return nullptr;
}

Place::Ptr Place::get_producing_port() const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_producing_port());
    }
    return nullptr;
}

Place::Ptr Place::get_input_port() const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_input_port());
    }
    return nullptr;
}

Place::Ptr Place::get_input_port(int input_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_input_port(input_port_index));
    }
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_input_port(input_name));
    }
    return nullptr;
}

Place::Ptr Place::get_input_port(const std::string& input_name, int input_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_input_port(input_name, input_port_index));
    }
    return nullptr;
}

Place::Ptr Place::get_output_port() const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_output_port());
    }
    return nullptr;
}

Place::Ptr Place::get_output_port(int output_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_output_port(output_port_index));
    }
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_output_port(output_name));
    }
    return nullptr;
}

Place::Ptr Place::get_output_port(const std::string& output_name, int output_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_output_port(output_name, output_port_index));
    }
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_ports() const {
    if (m_pimpl) {
        return transform_pimpls(m_pimpl->get_consuming_ports());
    }
    return {};
}

bool Place::is_input() const {
    if (m_pimpl) {
        return m_pimpl->is_input();
    }
    FRONT_END_NOT_IMPLEMENTED(is_input);
}

bool Place::is_output() const {
    if (m_pimpl) {
        return m_pimpl->is_output();
    }
    FRONT_END_NOT_IMPLEMENTED(is_output);
}

bool Place::is_equal(const Ptr& another) const {
    if (m_pimpl) {
        return m_pimpl->is_equal(another);
    }
    FRONT_END_NOT_IMPLEMENTED(is_equal);
}

bool Place::is_equal_data(const Ptr& another) const {
    if (m_pimpl) {
        return m_pimpl->is_equal_data(another);
    }
    FRONT_END_NOT_IMPLEMENTED(is_equal_data);
}

Place::Ptr Place::get_source_tensor() const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_source_tensor());
    }
    return nullptr;
}

Place::Ptr Place::get_source_tensor(int input_port_index) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_source_tensor(input_port_index));
    }
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName, int inputPortIndex) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_source_tensor(inputName, inputPortIndex));
    }
    return nullptr;
}

Place::Ptr Place::get_source_tensor(const std::string& inputName) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_source_tensor(inputName));
    }
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_target_tensor(outputPortName));
    }
    return nullptr;
}

Place::Ptr Place::get_target_tensor(const std::string& outputPortName, int outputPortIndex) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_target_tensor(outputPortName, outputPortIndex));
    }
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_producing_operation(inputName));
    }
    return nullptr;
}

Place::Ptr Place::get_producing_operation(const std::string& inputName, int inputPortIndex) const {
    if (m_pimpl) {
        return make_shared_w_pimpl(m_pimpl->get_producing_operation(inputName, inputPortIndex));
    }
    return nullptr;
}

std::vector<Place::Ptr> Place::get_consuming_operations(const std::string& outputPortName) const {
    if (m_pimpl) {
        return transform_pimpls(m_pimpl->get_consuming_operations(outputPortName));
    }
    return {};
}
