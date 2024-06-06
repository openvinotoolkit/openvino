// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/node.hpp"
#include "openvino/frontend/decoder.hpp"

namespace ov {
namespace frontend {
namespace jax {

class JaxDecoder : public IDecoder {
public:
    // TODO: is this really needed in jax?
    virtual OutputVector as_constant() const = 0;

    // Get string from constant. Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds
    // that fit
    virtual const std::string& as_string() const = 0;

    // Get the const variable.
    virtual Any const_var(size_t index) const = 0;

    virtual const std::string get_op_type() const = 0;

    virtual const std::vector<size_t>& inputs() const = 0;

    // Return whether the i th input has abstract value.
    virtual const bool input_has_aval(size_t index) const = 0;

    // Get the abstract value of the input, if any.
    virtual Any input_aval(size_t index) const = 0;

    virtual bool input_is_none(size_t index) const = 0;

    // Return debug name of the input tensor
    virtual const std::string& get_input_debug_name(size_t index) const = 0;

    // Return signature name of the input tensor
    virtual const std::string& get_input_signature_name(size_t index) const = 0;

    // Return the input shape
    virtual PartialShape get_input_shape(size_t index) const = 0;

    virtual Any get_input_type(size_t index) const = 0;

    // TODO: is this really needed in jax frontend?
    virtual size_t get_named_input(const std::string& name) const = 0;

    // Return debug name of the input tensor
    virtual const std::string& get_output_debug_name(size_t index) const = 0;

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

    // Get the debug information, which may contains some information from python.
    virtual const std::string debug_info() const = 0;

    // Embed mapping to/from the original node representation from/to node passed as a parameter
    // the representation of this mapping is specific for particular decorated type and may be NOP
    // returns the same node as syntactically convenient way to make nested sentences in code
    virtual std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const = 0;

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    // node_visitor is a function that will be fed by nodes in subgraph for all nodes in graph
    virtual void visit_subgraph(std::function<void(std::shared_ptr<JaxDecoder>)> node_visitor) const = 0;

    /// Returns named attribute as Any.
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    /// Returns the id of the decoder type.
    virtual const std::string& decoder_type_name() const = 0;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
