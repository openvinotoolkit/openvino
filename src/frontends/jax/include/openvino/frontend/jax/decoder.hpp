// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/decoder.hpp"

namespace ov {
namespace frontend {
namespace jax {

class JaxDecoder : public IDecoder {
public:
    virtual OutputVector as_constant() const = 0;

    virtual const std::string get_op_type() const = 0;

    virtual const std::vector<size_t>& inputs() const = 0;

    // Return signature name of the input tensor
    virtual const std::string& get_input_signature_name(size_t index) const = 0;

    // Return the input shape
    virtual PartialShape get_input_shape(size_t index) const = 0;

    virtual Any get_input_type(size_t index) const = 0;

    virtual size_t get_named_param(const std::string& name) const = 0;

    virtual OutputVector get_named_param_as_constant(const std::string& name) const = 0;

    virtual const std::vector<std::string>& get_param_names() const = 0;

    // Return name of the output tensor
    virtual const std::string& get_output_name(size_t index) const = 0;

    // Return the output shape
    virtual PartialShape get_output_shape(size_t index) const = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-specific data type object
    // (see custom_type.hpp)
    virtual Any get_output_type(size_t index) const = 0;

    // Get the inputs size. Note that jaxpr flattens the inputs in python. Therefore we do not need to deal with nested
    // inputs here.
    virtual size_t num_inputs() const = 0;

    // Get the outputs size.
    virtual size_t num_outputs() const = 0;

    // Return a vector of output IDs
    virtual const std::vector<size_t>& outputs() const = 0;

    virtual size_t output(size_t index) const = 0;

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    // node_visitor is a function that will be fed by nodes in subgraph for all nodes in graph
    virtual void visit_subgraph(std::function<void(std::shared_ptr<JaxDecoder>)> node_visitor) const = 0;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
