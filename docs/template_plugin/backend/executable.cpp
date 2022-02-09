// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executable.hpp"

#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::Executable::Executable() {}

runtime::Executable::~Executable() {}

bool runtime::Executable::call_with_validate(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                             const vector<shared_ptr<runtime::Tensor>>& inputs) {
    validate(outputs, inputs);
    return call(outputs, inputs);
}

void runtime::Executable::validate(const vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                   const vector<std::shared_ptr<runtime::Tensor>>& inputs) {
    const ParameterVector& parameters = get_parameters();
    const ResultVector& results = get_results();
    if (parameters.size() != inputs.size()) {
        stringstream ss;
        ss << "Call input count " << inputs.size() << " does not match Function's Parameter count "
           << parameters.size();
        throw runtime_error(ss.str());
    }
    if (results.size() != outputs.size()) {
        stringstream ss;
        ss << "Call output count " << outputs.size() << " does not match Function's Result count " << results.size();
        throw runtime_error(ss.str());
    }

    for (size_t i = 0; i < parameters.size(); i++) {
        if (parameters[i]->get_element_type().is_static() &&
            parameters[i]->get_element_type() != inputs[i]->get_element_type()) {
            stringstream ss;
            ss << "Input " << i << " type '" << inputs[i]->get_element_type() << "' does not match Parameter type '"
               << parameters[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        if (!(parameters[i]->get_output_partial_shape(0).relaxes(inputs[i]->get_partial_shape()))) {
            stringstream ss;
            ss << "Input " << i << " shape " << inputs[i]->get_partial_shape() << " does not match Parameter shape "
               << parameters[i]->get_output_partial_shape(0);
            throw runtime_error(ss.str());
        }
    }

    for (size_t i = 0; i < results.size(); i++) {
        if (outputs[i]->get_element_type().is_static() && results[i]->get_element_type().is_static() &&
            results[i]->get_element_type() != outputs[i]->get_element_type()) {
            stringstream ss;
            ss << "Output " << i << " type '" << outputs[i]->get_element_type() << "' does not match Result type '"
               << results[i]->get_element_type() << "'";
            throw runtime_error(ss.str());
        }
        if (!outputs[i]->get_partial_shape().relaxes(results[i]->get_output_partial_shape(0))) {
            stringstream ss;
            ss << "Output " << i << " shape " << outputs[i]->get_partial_shape() << " does not match max Result shape "
               << results[i]->get_output_partial_shape(0).get_max_shape();
            throw runtime_error(ss.str());
        }
    }
}

const ngraph::ParameterVector& runtime::Executable::get_parameters() const {
    return m_parameters;
}

const ngraph::ResultVector& runtime::Executable::get_results() const {
    return m_results;
}

size_t runtime::Executable::get_preferred_pipeline_depth() const {
    return 2;
}

void runtime::Executable::set_parameters_and_results(const Function& func) {
    m_parameters = func.get_parameters();
    m_results = func.get_results();
}

vector<runtime::PerformanceCounter> runtime::Executable::get_performance_data() const {
    return vector<PerformanceCounter>();
}

void runtime::Executable::save(std::ostream& /* output_stream */) {
    throw runtime_error("save operation unimplemented.");
}

shared_ptr<runtime::Tensor> runtime::Executable::create_input_tensor(size_t /* input_index */) {
    throw runtime_error("create_input_tensor unimplemented");
}

shared_ptr<runtime::Tensor> runtime::Executable::create_input_tensor(size_t /* input_index */,
                                                                     void* /* memory_pointer */) {
    throw runtime_error("create_input_tensor unimplemented");
}

shared_ptr<runtime::Tensor> runtime::Executable::create_output_tensor(size_t /* output_index */) {
    throw runtime_error("create_output_tensor unimplemented");
}

shared_ptr<runtime::Tensor> runtime::Executable::create_output_tensor(size_t /* output_index */,
                                                                      void* /* memory_pointer */) {
    throw runtime_error("create_output_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> runtime::Executable::create_input_tensor(size_t /* input_index */,
                                                                             size_t /* pipeline_depth */) {
    throw runtime_error("create_input_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> runtime::Executable::create_input_tensor(size_t /* input_index */,
                                                                             size_t /* pipeline_depth */,
                                                                             std::vector<void*> /* memory_pointer */) {
    throw runtime_error("create_input_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> runtime::Executable::create_output_tensor(size_t /* output_index */,
                                                                              size_t /* pipeline_depth */) {
    throw runtime_error("create_output_tensor unimplemented");
}

vector<shared_ptr<runtime::Tensor>> runtime::Executable::create_output_tensor(size_t /* output_index */,
                                                                              size_t /* pipeline_depth */,
                                                                              std::vector<void*> /* memory_pointer */) {
    throw runtime_error("create_output_tensor unimplemented");
}
