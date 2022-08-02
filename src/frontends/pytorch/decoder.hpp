#pragma once

#include <memory>

// TODO: rough!
#include "openvino/openvino.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

/// Plays a role of node, block and module decoder (kind of temporary fat API)
class Decoder : public std::enable_shared_from_this<Decoder> {      // TODO: Is it required to be enable_shared_from_this?
public:

    // Do not search for input in tensor map; try to access it as a constant of specified type T and return its value
    // Using Any here is an easy way to avoid template definition, returned object is supposed to be of one of the fundamental types like int, float etc.
    virtual Any const_input (size_t index) const = 0;

    // Using size_t for input/output unuque ids are in sync with torch code, see def in torch/include/torch/csrc/jit/ir/ir.h, Value::unique_

    // TODO: set of input and output methods are not aligned; also they are not aligned with the rest of FEs

    // Input tensor id
    virtual size_t input (size_t index) const = 0;

    virtual std::vector<size_t> inputs () const = 0;

    // TODO: required? can be implemented in the context of a single node?
    virtual bool input_is_none (size_t index) const = 0;

    // Work for natural constant nodes, e.g. for prim::Constant; don't know other nodes kinds that fit
    virtual OutputVector as_constant () = 0;

    // Returns PT node kind as a string mnemonics for native type uint32_t Symbol in Torch
    // Decide whether we need an equivalent member for integer representation (in this case a map is required to understand what it means)
    virtual std::string get_op_type() const = 0;

    // TODO: use canonical name output_size 
    virtual size_t num_of_outputs () const = 0;

    // Return a vector of output IDs
    virtual std::vector<size_t> outputs () const = 0;

    // Embed mapping to/from the original node representation from/to node passed as a parameter
    // the representation of this mapping is specific for particular decored type and may be NOP
    // returns the same node as syntactically convenient way to make nested sentences in code
    virtual std::shared_ptr<Node> mark_node (std::shared_ptr<Node> ov_node) const = 0;

    // Call mark_node for each node from the vector
    void mark_nodes (std::vector<std::shared_ptr<Node>> ov_nodes) const {
        for (auto& ov_node : ov_nodes) {
            mark_node(ov_node);
        }
    }

    // Syntactic sugar around mark_node -- just calls it for corresponding node for the passed output port
    Output<Node> mark_output (Output<Node> ov_output) const {
        mark_node(ov_output.get_node_shared_ptr());
        return ov_output;
    }

    /// \brief Returns the number of sub-graphs that can be enumerated with get_subgraph
    virtual size_t get_subgraph_size() const = 0;

    /// \brief Returns subgraph converted on demand by the first access
    /// If there is no query for specific sub-graph it shouldn't be converted
    /// idx should be in range 0..get_subgraph_size()-1
    // TODO: Why int for idx? Why not unsigned? Just reused the same type fro get_input
    virtual std::shared_ptr<Decoder> get_subgraph(int idx) const = 0;
};

}
}
}
