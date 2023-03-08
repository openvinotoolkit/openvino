// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/frontend/decoder.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

/// Plays a role of node, block and module decoder (kind of temporary fat API)
class TorchDecoder : public IDecoder {
public:
    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    // Using Any here is an easy way to avoid template definition, returned object is supposed to be of one of the
    // fundamental types like int, float etc.
    virtual Any const_input(size_t index) const = 0;

    // Using size_t for input/output unuque ids are in sync with torch code, see def in
    // torch/include/torch/csrc/jit/ir/ir.h, Value::unique_

    // TODO: set of input and output methods are not aligned; also they are not aligned with the rest of FEs

    virtual const std::vector<size_t>& inputs() const = 0;

    // ------------------------------
    // TODO: physically inputs and outputs refer to PT Values so shape/type is not a property of input/output
    // Do we need a separate Decoder for Tensor to request properties of it instead of having an impression
    // that inputs/outputs have types and shapes?

    // Return debug name of the input tensor
    virtual const std::string& get_input_debug_name(size_t index) const = 0;

    // Return shape if inputs has torch::Tensor type in the original model, otherwise returns the shape [] of a scalar
    virtual PartialShape get_input_shape(size_t index) const = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-specific data type object
    // (see custom_type.hpp)
    virtual Any get_input_type(size_t index) const = 0;

    // TODO: Consider deleting this method, probably it doesn't make sence outside Torch JIT execution
    virtual const std::vector<size_t>& get_input_transpose_order(size_t index) const = 0;

    // Return debug name of the input tensor
    virtual const std::string& get_output_debug_name(size_t index) const = 0;

    // Return shape if inputs has torch::Tensor type in the original model, otherwise returns the shape [] of a scalar
    virtual PartialShape get_output_shape(size_t index) const = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-specific data type object
    // (see custom_type.hpp)
    virtual Any get_output_type(size_t index) const = 0;

    // TODO: Consider deleting this method, probably it doesn't make sence outside Torch JIT execution
    virtual const std::vector<size_t>& get_output_transpose_order(size_t index) const = 0;
    // ------------------------------

    // TODO: required? can be implemented in the context of a single node?
    virtual bool input_is_none(size_t index) const = 0;

    virtual OutputVector try_decode_get_attr() const = 0;

    // Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds that fit
    // TODO: why OutputVector instead of just single output?
    virtual OutputVector as_constant() const = 0;

    // Get string from constant. Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds
    // that fit
    virtual const std::string& as_string() const = 0;

    // Returns PT node kind as a string mnemonics for native type uint32_t Symbol in Torch
    // Decide whether we need an equivalent member for integer representation (in this case a map is required to
    // understand what it means)
    virtual const std::string& get_op_type() const = 0;

    // Returns PT node schema as a string
    virtual const std::string& get_schema() const = 0;

    // TODO: use canonical name output_size
    virtual size_t num_of_outputs() const = 0;

    // Return a vector of output IDs
    virtual const std::vector<size_t>& outputs() const = 0;

    // Return a vector of output IDs
    virtual size_t output(size_t index) const = 0;

    // Embed mapping to/from the original node representation from/to node passed as a parameter
    // the representation of this mapping is specific for particular decored type and may be NOP
    // returns the same node as syntactically convenient way to make nested sentences in code
    virtual std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const = 0;

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    virtual size_t get_subgraph_size() const = 0;

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    // node_visitor is a function that will be fed by nodes in subgraph for all nodes in graph
    virtual void visit_subgraph(std::function<void(std::shared_ptr<TorchDecoder>)> node_visitor) const = 0;

    /// Probably this toghether with immediate nodes visitor is a replacement for visit_subgraphs with an index
    virtual std::shared_ptr<TorchDecoder> get_subgraph_decoder(size_t index) const = 0;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
