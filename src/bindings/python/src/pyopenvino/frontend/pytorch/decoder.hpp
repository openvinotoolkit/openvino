// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>


#include "openvino/frontend/pytorch/decoder.hpp"

namespace py = pybind11;

    using ov::Any;
    using ov::PartialShape;
    using ov::OutputVector;
    using ov::Node;

/// Trampoline class to support inheritence from Decoder in Python
    class PyDecoder : public ov::frontend::pytorch::Decoder {


    using ov::frontend::pytorch::Decoder::Decoder;
        
    Any const_input (size_t index) const override
    { PYBIND11_OVERRIDE_PURE(Any, Decoder, const_input, index); }

    size_t input (size_t index) const override
    { PYBIND11_OVERRIDE_PURE(size_t, Decoder, input, index); }

    std::vector<size_t> inputs () const override
    { PYBIND11_OVERRIDE_PURE(std::vector<size_t>, ov::frontend::pytorch::Decoder, inputs); }

    PartialShape get_input_shape (size_t index) override
    { PYBIND11_OVERRIDE_PURE(PartialShape, Decoder, get_input_shape, index); }

    Any get_input_type (size_t index) override
    { PYBIND11_OVERRIDE_PURE(Any, Decoder, get_input_type, index); }

    std::vector<size_t> get_input_transpose_order (size_t index) override
    { PYBIND11_OVERRIDE_PURE(std::vector<size_t>, Decoder, get_input_transpose_order, index); }

    std::vector<size_t> get_output_transpose_order (size_t index) override
    { PYBIND11_OVERRIDE_PURE(std::vector<size_t>, Decoder, get_output_transpose_order, index); }

    PartialShape get_output_shape (size_t index) override
    { PYBIND11_OVERRIDE_PURE(PartialShape, Decoder, get_output_shape, index); }

    Any get_output_type (size_t index) override
    { PYBIND11_OVERRIDE_PURE(Any, Decoder, get_output_type, index); }

    bool input_is_none (size_t index) const override
    { PYBIND11_OVERRIDE_PURE(bool, Decoder, input_is_none, index); }

    OutputVector as_constant () override
    { PYBIND11_OVERRIDE_PURE(OutputVector, Decoder, as_constant); }

    std::string as_string () override
    { PYBIND11_OVERRIDE_PURE(std::string, Decoder, as_string); }

    std::string get_op_type() const override
    { PYBIND11_OVERRIDE_PURE(std::string, Decoder, get_op_type); }

    std::string get_schema() const override
    { PYBIND11_OVERRIDE_PURE(std::string, Decoder, get_schema); }

    size_t num_of_outputs () const override
    { PYBIND11_OVERRIDE_PURE(size_t, Decoder, num_of_outputs); }

    std::vector<size_t> outputs () const override
    { PYBIND11_OVERRIDE_PURE(std::vector<size_t>, Decoder, outputs); }
    
    size_t output (size_t index) const override
    { PYBIND11_OVERRIDE_PURE(size_t, Decoder, output, index); }

    std::shared_ptr<Node> mark_node (std::shared_ptr<Node> ov_node) const override
    { PYBIND11_OVERRIDE_PURE(std::shared_ptr<Node>, Decoder, mark_node, ov_node); }

    size_t get_subgraph_size() const override
    { PYBIND11_OVERRIDE_PURE(size_t, Decoder, get_subgraph_size); }

    void visit_subgraph(std::function<void(std::shared_ptr<Decoder>)> node_visitor) const override
    { PYBIND11_OVERRIDE_PURE(void, Decoder, visit_subgraph, node_visitor); }

    std::shared_ptr<Decoder> get_subgraph_decoder (size_t index) const override
    { PYBIND11_OVERRIDE_PURE(std::shared_ptr<Decoder>, Decoder, get_subgraph_decoder, index); }

    void debug () const override
    { PYBIND11_OVERRIDE_PURE(void, Decoder, debug); }
};

void regclass_frontend_pytorch_decoder(py::module m);