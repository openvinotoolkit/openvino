// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "openvino/frontend/tensorflow/decoder.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritance from GraphIterator in Python
class PyDecoderBase : public ov::frontend::tensorflow::DecoderBase {

    ov::Any get_attribute(const std::string &name) const override{
        PYBIND11_OVERRIDE_PURE(ov::Any, DecoderBase, get_attribute, name);
    }

    size_t get_input_size() const override{
        PYBIND11_OVERRIDE_PURE(size_t, DecoderBase, get_input_size);
    }


    std::string get_input_node_name(size_t input_port_idx) const {
        PYBIND11_OVERRIDE_PURE(std::string, DecoderBase, get_input_node_name, input_port_idx);
    }

    size_t get_input_node_name_output_port_index(size_t input_port_idx) const {
        PYBIND11_OVERRIDE_PURE(size_t, DecoderBase, get_input_node_name_output_port_index, input_port_idx);
    }

    std::string get_input_node_name_output_port_name(size_t input_port_idx) const {
        PYBIND11_OVERRIDE_PURE(std::string, DecoderBase, get_input_node_name_output_port_name, input_port_idx);
    }

    void get_input_node(size_t input_port_idx,
                                std::string &producer_name,
                                std::string &producer_output_port_name,
                                size_t &producer_output_port_index) const override{
        producer_name = get_input_node_name(input_port_idx);
        producer_output_port_index = get_input_node_name_output_port_index(input_port_idx);
        producer_output_port_name = get_input_node_name_output_port_name(input_port_idx);
    }

    const std::string &get_op_type() const override{
        PYBIND11_OVERRIDE_PURE(std::string&, DecoderBase, get_op_type);
    }

    const std::string &get_op_name() const override{
        PYBIND11_OVERRIDE_PURE(std::string&, DecoderBase, get_op_name);
    }
};

void regclass_frontend_tensorflow_decoder_base(py::module m);