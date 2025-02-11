// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/frontend/jax/decoder.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritence from JaxDecoder in Python
class PyDecoder : public ov::frontend::jax::JaxDecoder {
    using ov::frontend::jax::JaxDecoder::JaxDecoder;

    ov::OutputVector as_constant() const override {
        PYBIND11_OVERRIDE_PURE(ov::OutputVector, JaxDecoder, as_constant);
    }

    const std::string get_op_type() const override {
        PYBIND11_OVERRIDE_PURE(std::string, JaxDecoder, get_op_type);
    }

    const std::string& get_input_signature_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, get_input_signature_name, index);
    }

    ov::PartialShape get_input_shape(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::PartialShape, JaxDecoder, get_input_shape, index);
    }

    ov::PartialShape get_output_shape(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::PartialShape, JaxDecoder, get_output_shape, index);
    }

    ov::Any get_input_type(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, JaxDecoder, get_input_type, index);
    }

    size_t get_named_param(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(size_t, JaxDecoder, get_named_param, name);
    }

    ov::OutputVector get_named_param_as_constant(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(ov::OutputVector, JaxDecoder, get_named_param_as_constant, name);
    }

    const std::vector<std::string>& get_param_names() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<std::string>&, JaxDecoder, get_param_names);
    }

    const std::string& get_output_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, get_output_name, index);
    }

    ov::Any get_output_type(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, JaxDecoder, get_output_type, index);
    }

    const std::vector<size_t>& inputs() const override {
        PYBIND11_OVERRIDE_PURE(std::vector<size_t>&, JaxDecoder, inputs);
    }

    size_t num_inputs() const override {
        PYBIND11_OVERRIDE_PURE(size_t, JaxDecoder, num_inputs);
    }

    size_t num_outputs() const override {
        PYBIND11_OVERRIDE_PURE(size_t, JaxDecoder, num_outputs);
    }

    size_t output(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(size_t, JaxDecoder, output, index);
    }

    const std::vector<size_t>& outputs() const override {
        PYBIND11_OVERRIDE_PURE(std::vector<size_t>&, JaxDecoder, outputs);
    }

    void visit_subgraph(std::function<void(std::shared_ptr<JaxDecoder>)> node_visitor) const override {
        PYBIND11_OVERRIDE_PURE(void, JaxDecoder, visit_subgraph, node_visitor);
    }
};

void regclass_frontend_jax_decoder(py::module m);
