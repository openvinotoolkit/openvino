// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "symbol_table.hpp"

#include "node.h"
#include "openvino/core/dimension.hpp"

namespace ov::intel_cpu {

size_t SymbolTable::register_symbol(const std::shared_ptr<ov::Symbol>& sym) {
    if (!sym) {
        // Should not happen — caller should check before calling
        OPENVINO_THROW("SymbolTable::register_symbol called with nullptr");
    }

    if (sym->is_leaf()) {
        auto root = ov::symbol::ancestor_of(sym);
        auto it = m_index.find(root.get());
        if (it != m_index.end()) {
            return it->second;
        }
        size_t idx = m_symbols.size();
        m_symbols.push_back(root);
        m_index[root.get()] = idx;
        return idx;
    }

    // Compound symbol: recursively register operands first (ensures topological order)
    register_symbol(sym->get_lhs());
    register_symbol(sym->get_rhs());

    auto it = m_index.find(sym.get());
    if (it != m_index.end()) {
        return it->second;
    }
    size_t idx = m_symbols.size();
    m_symbols.push_back(sym);
    m_index[sym.get()] = idx;
    return idx;
}

void SymbolTable::build(const std::vector<NodePtr>& graphNodes) {
    m_symbols.clear();
    m_index.clear();

    // Pass 1: collect all symbols from input and output partial shapes
    for (const auto& node : graphNodes) {
        for (const auto& pshape : node->originalInputShapes) {
            for (const auto& dim : pshape) {
                if (dim.get_symbol()) {
                    register_symbol(dim.get_symbol());
                }
            }
        }
        for (const auto& pshape : node->originalOutputShapes) {
            for (const auto& dim : pshape) {
                if (dim.get_symbol()) {
                    register_symbol(dim.get_symbol());
                }
            }
        }
    }

    // Allocate runtime values buffer
    m_runtime_values.resize(m_symbols.size(), 0);
}

void SymbolTable::resolve_inputs(const std::vector<NodePtr>& inputNodes) {
    // Step 1: populate leaf symbols from actual input tensor dimensions
    for (const auto& node : inputNodes) {
        if (node->originalOutputShapes.empty()) {
            continue;
        }
        const auto& pshape = node->originalOutputShapes[0];

        // Get actual dims from the child edge memory
        auto edge = node->getChildEdgeAt(0);
        if (!edge || !edge->getMemoryPtr()) {
            continue;
        }
        const auto& actualDims = edge->getMemory().getStaticDims();

        for (size_t d = 0; d < pshape.size() && d < actualDims.size(); ++d) {
            std::shared_ptr<ov::Symbol> sym = pshape[d].get_symbol();
            if (!sym) {
                continue;
            }
            auto root = ov::symbol::ancestor_of(sym);
            auto it = m_index.find(root.get());
            if (it != m_index.end()) {
                // std::cout << "Leaf symbol " << sym << " resolved to " << actualDims[d] << std::endl;
                m_runtime_values[it->second] = static_cast<int64_t>(actualDims[d]);
            }
        }
    }

    // Step 2: forward pass to evaluate compound symbols
    for (size_t i = 0; i < m_symbols.size(); ++i) {
        const auto& sym = m_symbols[i];
        if (sym->is_leaf()) {
            continue;  // already populated from step 1
        }

        // @todo claude: add SUB support when SymbolKind::SUB is added
        auto lhs_it = m_index.find(sym->get_lhs().get());
        auto rhs_it = m_index.find(sym->get_rhs().get());
        OPENVINO_ASSERT(lhs_it != m_index.end() && rhs_it != m_index.end(),
                        "SymbolTable: compound symbol operands not found in index");

        int64_t lhs_val = m_runtime_values[lhs_it->second];
        int64_t rhs_val = m_runtime_values[rhs_it->second];

        switch (sym->get_kind()) {
        case ov::SymbolKind::ADD:
            m_runtime_values[i] = lhs_val + rhs_val;
            break;
        case ov::SymbolKind::MUL:
            m_runtime_values[i] = lhs_val * rhs_val;
            break;
        default:
            OPENVINO_THROW("SymbolTable: unsupported compound symbol kind");
        }
        // std::cout << "Compound symbol " << sym << " resolved to " << m_runtime_values[i] << std::endl;
    }
}

VectorDims SymbolTable::resolve_shape(const SymShape& shape) const {
    VectorDims result;
    result.reserve(shape.size());
    for (auto dim : shape) {
        result.push_back(static_cast<Dim>(resolve(dim)));
    }
    return result;
}

SymShape SymbolTable::to_sym_shape(const ov::PartialShape& pshape) const {
    SymShape result;
    result.reserve(pshape.size());
    for (const auto& dim : pshape) {
        if (dim.is_static()) {
            result.push_back(static_cast<int64_t>(dim.get_length()));
        } else if (auto sym = dim.get_symbol()) {
            // Look up the symbol — for leaves use root pointer, for compounds use direct pointer
            ov::Symbol* lookup_ptr = sym->is_leaf() ? ov::symbol::ancestor_of(sym).get() : sym.get();
            auto it = m_index.find(lookup_ptr);
            if (it != m_index.end()) {
                result.push_back(~static_cast<int64_t>(it->second));
            } else {
                result.push_back(SYMDIM_UNDEFINED);
            }
        } else {
            result.push_back(SYMDIM_UNDEFINED);
        }
    }
    return result;
}

}  // namespace ov::intel_cpu
