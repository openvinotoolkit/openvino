// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cpu_types.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/symbol.hpp"

namespace ov::intel_cpu {

class Node;
using NodePtr = std::shared_ptr<Node>;

/// Symbolic dimension encoding:
///   value >= 0  ->  static dimension value
///   value < 0   ->  ~value is index into SymbolTable (bitwise NOT)
using SymDim = int64_t;
using SymShape = std::vector<SymDim>;

/// Sentinel value: dynamic dimension without a symbol — forces fallback to shapeInfer
static constexpr SymDim SYMDIM_UNDEFINED = std::numeric_limits<int64_t>::min();

/// SymbolTable maps ov::Symbol pointers to compact integer indices and
/// resolves their runtime values from actual input shapes.
class SymbolTable {
public:
    /// Walk all graph nodes, collect symbols from originalInput/OutputShapes,
    /// and register them (leaves first, then compounds).
    void build(const std::vector<NodePtr>& graphNodes);

    /// Populate runtime values for leaf symbols from actual input tensor shapes,
    /// then forward-evaluate compound symbols.
    void resolve_inputs(const std::vector<NodePtr>& inputNodes);

    /// Resolve a single SymDim to a concrete value.
    int64_t resolve(SymDim dim) const {
        return dim >= 0 ? dim : m_runtime_values[~dim];
    }

    /// Resolve a full SymShape into VectorDims.
    VectorDims resolve_shape(const SymShape& shape) const;

    /// Convert an ov::PartialShape to a SymShape using the registered table.
    /// Static dims keep their value; dynamic dims with a registered symbol get ~index;
    /// dynamic dims without a symbol get SYMDIM_UNDEFINED.
    SymShape to_sym_shape(const ov::PartialShape& pshape) const;

    bool empty() const {
        return m_symbols.empty();
    }
    size_t size() const {
        return m_symbols.size();
    }

private:
    /// Register a symbol (and its operands if compound). Returns the table index.
    size_t register_symbol(const std::shared_ptr<ov::Symbol>& sym);

    // Unique symbols, topologically ordered (operands before compounds)
    std::vector<std::shared_ptr<ov::Symbol>> m_symbols;

    // Maps ov::Symbol raw pointer -> table index
    // For leaves: uses the root (ancestor) pointer
    // For compounds: uses the direct pointer
    std::unordered_map<ov::Symbol*, size_t> m_index;

    // Runtime values, filled per-inference by resolve_inputs()
    std::vector<int64_t> m_runtime_values;
};

}  // namespace ov::intel_cpu
