// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/frontend/decoder.hpp"
#include "openvino/frontend/tensorflow/graph_iterator.hpp"

namespace py = pybind11;

/// Trampoline class to support inheritance from GraphIterator in Python
class PyGraphIterator : public ov::frontend::tensorflow::GraphIterator {
    /* Inherit the constructors */
    using ov::frontend::tensorflow::GraphIterator::GraphIterator;
    using map_str_to_str = std::map<std::string, std::string>;

    /// \brief Get a number of operation nodes in the graph
    size_t size() const override {
        PYBIND11_OVERRIDE_PURE(size_t, GraphIterator, size);
    }

    /// \brief Set iterator to the start position
    void reset() override {
        PYBIND11_OVERRIDE_PURE(void, GraphIterator, reset);
    }

    /// \brief Move to the next node in the graph
    void next() override {
        next_impl();
    }

    /// Implementation of next method, it is needed to be in separate method to avoid shadowing of Python "next"
    /// operator.
    void next_impl() {
        PYBIND11_OVERRIDE_PURE(void, GraphIterator, next_impl);
    }

    /// \brief Returns true if iterator goes out of the range of available nodes
    bool is_end() const override {
        PYBIND11_OVERRIDE_PURE(bool, GraphIterator, is_end);
    }

    /// \brief Return a pointer to a decoder of the current node
    std::shared_ptr<ov::frontend::DecoderBase> get_decoder() const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<ov::frontend::DecoderBase>, GraphIterator, get_decoder);
    }

    /// \brief Checks if the main model graph contains a function of the requested name in the library
    /// Returns GraphIterator to this function and nullptr, if it does not exist
    std::shared_ptr<GraphIterator> get_body_graph_iterator(const std::string& func_name) const override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<GraphIterator>, GraphIterator, get_body_graph_iterator, func_name);
    }

    /// \brief Returns a vector of input names in the original order
    std::vector<std::string> get_input_names() const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, GraphIterator, get_input_names);
    }

    /// \brief Returns a vector of output names in the original order
    std::vector<std::string> get_output_names() const override {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>, GraphIterator, get_output_names);
    }

    /// \brief Returns a map from internal tensor name to (user-defined) external name for inputs
    map_str_to_str get_input_names_map() const override {
        PYBIND11_OVERRIDE_PURE(map_str_to_str, GraphIterator, get_input_names_map);
    }

    /// \brief Returns a map from internal tensor name to (user-defined) external name for outputs
    map_str_to_str get_output_names_map() const override {
        PYBIND11_OVERRIDE_PURE(map_str_to_str, GraphIterator, get_output_names_map);
    }
};

void regclass_frontend_tensorflow_graph_iterator(py::module m);