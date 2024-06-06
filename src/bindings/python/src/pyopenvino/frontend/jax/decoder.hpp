// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/jax/decoder.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritence from TorchDecoder in Python
class PyDecoder : public ov::frontend::jax::JaxDecoder {
    using ov::frontend::jax::JaxDecoder::JaxDecoder;

    ov::OutputVector as_constant() const override {
        PYBIND11_OVERRIDE_PURE(ov::OutputVector, JaxDecoder, as_constant);
    }

    const std::string& as_string() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, as_string);
    }

    ov::Any const_var(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, JaxDecoder, const_var, index);
    }

    const std::string get_op_type() const override {
        PYBIND11_OVERRIDE_PURE(std::string, JaxDecoder, get_op_type);
    }

    const std::string& get_input_debug_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, get_input_debug_name, index);
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

    size_t get_named_input(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(size_t, JaxDecoder, get_named_input, name);
    }

    const std::string& get_output_debug_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, get_output_debug_name, index);
    }

    ov::Any get_output_type(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, JaxDecoder, get_output_type, index);
    }

    const std::vector<size_t>& inputs() const override {
        PYBIND11_OVERRIDE_PURE(std::vector<size_t>&, JaxDecoder, inputs);
    }

    const bool input_has_aval(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(bool, JaxDecoder, input_has_aval, index);
    }

    ov::Any input_aval(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, JaxDecoder, input_aval, index);
    }

    bool input_is_none(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(bool, JaxDecoder, input_is_none, index);
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

    const std::string debug_info() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, debug_info);
    }

    std::shared_ptr<ov::Node> mark_node(std::shared_ptr<ov::Node> ov_node) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ov::Node>, JaxDecoder, mark_node, ov_node);
    }

    std::size_t get_subgraph_size() const override {
        PYBIND11_OVERRIDE_PURE(std::size_t, JaxDecoder, get_subgraph_size);
    }

    void visit_subgraph(std::function<void(std::shared_ptr<JaxDecoder>)> node_visitor) const override {
        PYBIND11_OVERRIDE_PURE(void, JaxDecoder, visit_subgraph, node_visitor);
    }

    std::shared_ptr<JaxDecoder> get_subgraph_decoder(std::size_t index) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<JaxDecoder>, JaxDecoder, get_subgraph_decoder, index);
    }

    ov::Any get_attribute(const std::string& name) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, JaxDecoder, get_attribute, name);
    }

    const std::string& decoder_type_name() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, JaxDecoder, decoder_type_name);
    }
};

void regclass_frontend_jax_decoder(py::module m);
