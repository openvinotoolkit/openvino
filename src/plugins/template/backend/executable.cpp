// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executable.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

ov::runtime::Executable::Executable() {}

ov::runtime::Executable::~Executable() {}

bool ov::runtime::Executable::call_with_validate(std::vector<ov::Tensor>& outputs,
                                                 const std::vector<ov::Tensor>& inputs) {
    validate(outputs, inputs);
    return call(outputs, inputs);
}

void ov::runtime::Executable::validate(const std::vector<ov::Tensor>& outputs, const std::vector<ov::Tensor>& inputs) {
    const ParameterVector& parameters = get_parameters();
    const ResultVector& results = get_results();
    if (parameters.size() != inputs.size()) {
        std::stringstream ss;
        ss << "Call input count " << inputs.size() << " does not match Function's Parameter count "
           << parameters.size();
        throw std::runtime_error(ss.str());
    }
    if (results.size() != outputs.size()) {
        std::stringstream ss;
        ss << "Call output count " << outputs.size() << " does not match Function's Result count " << results.size();
        throw std::runtime_error(ss.str());
    }

    for (size_t i = 0; i < parameters.size(); i++) {
        if (parameters[i]->get_element_type().is_static() &&
            parameters[i]->get_element_type() != inputs[i].get_element_type()) {
            std::stringstream ss;
            ss << "Input " << i << " type '" << inputs[i].get_element_type() << "' does not match Parameter type '"
               << parameters[i]->get_element_type() << "'";
            throw std::runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < results.size(); i++) {
        if (outputs[i].get_element_type().is_static() && results[i]->get_element_type().is_static() &&
            results[i]->get_element_type() != outputs[i].get_element_type()) {
            std::stringstream ss;
            ss << "Output " << i << " type '" << outputs[i].get_element_type() << "' does not match Result type '"
               << results[i]->get_element_type() << "'";
            throw std::runtime_error(ss.str());
        }
    }
}

const ov::ParameterVector& ov::runtime::Executable::get_parameters() const {
    return m_parameters;
}

const ov::ResultVector& ov::runtime::Executable::get_results() const {
    return m_results;
}

void ov::runtime::Executable::set_parameters_and_results(const ov::Model& model) {
    m_parameters = model.get_parameters();
    m_results = model.get_results();
}

ov::Tensor ov::runtime::Executable::create_input_tensor(size_t /* input_index */) {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Tensor ov::runtime::Executable::create_output_tensor(size_t /* output_index */) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::Tensor> ov::runtime::Executable::create_input_tensor(size_t /* input_index */,
                                                                     size_t /* pipeline_depth */) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ov::Tensor> ov::runtime::Executable::create_output_tensor(size_t /* output_index */,
                                                                      size_t /* pipeline_depth */) {
    OPENVINO_NOT_IMPLEMENTED;
}
