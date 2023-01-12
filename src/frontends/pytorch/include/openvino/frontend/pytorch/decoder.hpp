// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>

// TODO: rough!
#include "openvino/openvino.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

// Extendable type system which reflects TorchScript supported python data types
// Type nestings are built with the help of ov::Any

namespace type {

struct Tensor {
    Tensor() = default;
    explicit Tensor(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Tuple;

struct List {
    List() = default;

    // Specifies list of elements of element_type type, all elements have the same given type
    explicit List(const Any& _element_type) : element_type(_element_type) {}
    Any element_type;
};

struct Str {};

struct Optional;
struct Dict;
struct NamedTuple;
struct Union;

inline void print(const Any& x) {
    std::cout << "XDecoder.print: {" << x.type_info().name() << "}: ";
    if (x.is<element::Type>()) {
        std::cout << x.as<element::Type>();
    } else if (x.is<Tensor>()) {
        std::cout << "Tensor[";
        print(x.as<Tensor>().element_type);
        std::cout << "]";
    } else if (x.is<List>()) {
        std::cout << "List[";
        print(x.as<List>().element_type);
        std::cout << "]";
    } else {
        std::cout << "UNKNWON_ANY_TYPE";
    }
    std::cout << std::flush;
}

}  // namespace type

/// Plays a role of node, block and module decoder (kind of temporary fat API)
struct Decoder {  // TODO: Is it required to be enable_shared_from_this?
public:
    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    // Using Any here is an easy way to avoid template definition, returned object is supposed to be of one of the
    // fundamental types like int, float etc.
    virtual Any const_input(size_t index) const = 0;

    // Using size_t for input/output unuque ids are in sync with torch code, see def in
    // torch/include/torch/csrc/jit/ir/ir.h, Value::unique_

    // TODO: set of input and output methods are not aligned; also they are not aligned with the rest of FEs

    // Input tensor id
    virtual size_t input(size_t index) const = 0;

    virtual std::vector<size_t> inputs() const = 0;

    // ------------------------------
    // TODO: physically inputs and outputs refer to PT Values so shape/type is not a property of input/output
    // Do we need a separate Decoder for Tensor to request properties of it instead of having an impression
    // that inputs/outputs have types and shapes?

    // Return shape if inputs has torch::Tensor type in the original model, otherwise returns the shape [] of a scalar
    virtual PartialShape get_input_shape(size_t index) = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-sepcific data type object
    // (see custom_type.hpp)
    virtual Any get_input_type(size_t index) = 0;

    // TODO: Consider deleting this method, probably it doesn't make sence outside Torch JIT execution
    virtual std::vector<size_t> get_input_transpose_order(size_t index) = 0;

    // TODO: Consider deleting this method, probably it doesn't make sence outside Torch JIT execution
    virtual std::vector<size_t> get_output_transpose_order(size_t index) = 0;

    // Return shape if inputs has torch::Tensor type in the original model, otherwise returns the shape [] of a scalar
    virtual PartialShape get_output_shape(size_t index) = 0;

    // Return element::Type when it the original type can be represented, otherwise returns PT-sepcific data type object
    // (see custom_type.hpp)
    virtual Any get_output_type(size_t index) = 0;
    // ------------------------------

    // TODO: required? can be implemented in the context of a single node?
    virtual bool input_is_none(size_t index) const = 0;

    virtual ov::OutputVector try_decode_get_attr() = 0;

    // Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds that fit
    // TODO: why OutputVector instead of just single output?
    virtual OutputVector as_constant() = 0;

    // Get string from constant. Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds
    // that fit
    virtual std::string as_string() = 0;

    // Returns PT node kind as a string mnemonics for native type uint32_t Symbol in Torch
    // Decide whether we need an equivalent member for integer representation (in this case a map is required to
    // understand what it means)
    virtual std::string get_op_type() const = 0;

    // Returns PT node schema as a string
    virtual std::string get_schema() const = 0;

    // TODO: use canonical name output_size
    virtual size_t num_of_outputs() const = 0;

    // Return a vector of output IDs
    virtual std::vector<size_t> outputs() const = 0;

    // Return a vector of output IDs
    virtual size_t output(size_t index) const = 0;

    // Embed mapping to/from the original node representation from/to node passed as a parameter
    // the representation of this mapping is specific for particular decored type and may be NOP
    // returns the same node as syntactically convenient way to make nested sentences in code
    virtual std::shared_ptr<Node> mark_node(std::shared_ptr<Node> ov_node) const = 0;

    // Call mark_node for each node from the vector
    void mark_nodes(std::vector<std::shared_ptr<Node>> ov_nodes) const {
        for (auto& ov_node : ov_nodes) {
            mark_node(ov_node);
        }
    }

    // Syntactic sugar around mark_node -- just calls it for corresponding node for the passed output port
    Output<Node> mark_output(Output<Node> ov_output) const {
        mark_node(ov_output.get_node_shared_ptr());
        return ov_output;
    }

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    virtual size_t get_subgraph_size() const = 0;

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    // node_visitor is a function that will be fed by nodes in subgraph for all nodes in graph
    virtual void visit_subgraph(std::function<void(std::shared_ptr<Decoder>)> node_visitor) const = 0;

    /// Probably this toghether with immediate nodes visitor is a replacement for visit_subgraphs with an index
    virtual std::shared_ptr<Decoder> get_subgraph_decoder(size_t index) const = 0;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
