// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/frontend/pytorch/decoder.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritence from TorchDecoder in Python
class PyDecoder : public ov::frontend::pytorch::TorchDecoder {
    using ov::frontend::pytorch::TorchDecoder::TorchDecoder;

    ov::Any const_input(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, TorchDecoder, const_input, index);
    }

    const std::vector<size_t>& inputs() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, TorchDecoder, inputs);
    }

    const std::string& get_input_debug_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, get_input_debug_name, index);
    }

    const std::string& get_input_signature_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, get_input_signature_name, index);
    }

    ov::PartialShape get_input_shape(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::PartialShape, TorchDecoder, get_input_shape, index);
    }

    const std::vector<size_t>& get_input_strides(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, TorchDecoder, get_input_strides, index);
    }

    ov::Any get_input_type(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, TorchDecoder, get_input_type, index);
    }

    const std::string& get_output_debug_name(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, get_output_debug_name, index);
    }

    ov::PartialShape get_output_shape(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::PartialShape, TorchDecoder, get_output_shape, index);
    }

    ov::Any get_output_type(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(ov::Any, TorchDecoder, get_output_type, index);
    }

    bool input_is_none(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(bool, TorchDecoder, input_is_none, index);
    }

    ov::OutputVector try_decode_get_attr() const override {
        PYBIND11_OVERRIDE_PURE(ov::OutputVector, TorchDecoder, try_decode_get_attr);
    }

    ov::OutputVector as_constant() const override {
        PYBIND11_OVERRIDE_PURE(ov::OutputVector, TorchDecoder, as_constant);
    }

    const std::string& as_string() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, as_string);
    }

    const std::string& get_op_type() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, get_op_type);
    }

    const std::string& get_schema() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, get_schema);
    }

    size_t num_of_outputs() const override {
        PYBIND11_OVERRIDE_PURE(size_t, TorchDecoder, num_of_outputs);
    }

    size_t output_list_size() const override {
        PYBIND11_OVERRIDE_PURE(size_t, TorchDecoder, output_list_size);
    }

    const std::vector<size_t>& outputs() const override {
        PYBIND11_OVERRIDE_PURE(const std::vector<size_t>&, TorchDecoder, outputs);
    }

    size_t output(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(size_t, TorchDecoder, output, index);
    }

    std::shared_ptr<ov::Node> mark_node(std::shared_ptr<ov::Node> ov_node) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ov::Node>, TorchDecoder, mark_node, ov_node);
    }

    size_t get_subgraph_size() const override {
        PYBIND11_OVERRIDE_PURE(size_t, TorchDecoder, get_subgraph_size);
    }

    void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)> node_visitor) const override {
        PYBIND11_OVERRIDE_PURE(void, TorchDecoder, visit_subgraph, node_visitor);
    }

    std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<TorchDecoder>, TorchDecoder, get_subgraph_decoder, index);
    }

    bool may_produce_alias(size_t in_index, size_t out_index) const override {
        PYBIND11_OVERRIDE_PURE(bool, TorchDecoder, may_produce_alias, in_index, out_index);
    }

    bool is_input_inlined(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(bool, TorchDecoder, is_input_inlined, index);
    }

    std::shared_ptr<TorchDecoder> get_inlined_input_decoder(size_t index) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<TorchDecoder>, TorchDecoder, get_inlined_input_decoder, index);
    }

    ov::Any get_attribute(const std::string &name) const override{
        PYBIND11_OVERRIDE_PURE(ov::Any, TorchDecoder, get_attribute, name);
    }

    size_t get_named_input(const std::string &name) const override{
        PYBIND11_OVERRIDE_PURE(size_t, TorchDecoder, get_named_input, name);
    }

    const std::string& decoder_type_name() const override {
        PYBIND11_OVERRIDE_PURE(const std::string&, TorchDecoder, decoder_type_name);
    }

    ov::frontend::pytorch::DecoderRTInfo get_rt_info() const override {
        PYBIND11_OVERRIDE_PURE(ov::frontend::pytorch::DecoderRTInfo, TorchDecoder, get_rt_info);
    }
};

void regclass_frontend_pytorch_decoder(py::module m);
