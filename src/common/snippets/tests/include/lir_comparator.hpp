// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/graph_comparator.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/lowered/specific_loop_iter_handlers.hpp"

namespace ov {
namespace test {
namespace snippets {
class LIRComparator {
public:
    using NodesCmpValues = FunctionsComparator::CmpValues;
    using Result = FunctionsComparator::Result;

    enum LIRCmpValues {
        NONE = 0,
        PORT_DESCRIPTORS = 1 << 0,
        PORT_CONNECTORS = 1 << 1,
        LOOP_INDICES = 1 << 2,
        LOOP_MANAGER = 1 << 3,
    };

    // Creates LIRComparator with all CmpValues disabled
    static LIRComparator no_default() noexcept {
        return LIRComparator(NodesCmpValues::NONE, LIRCmpValues::NONE);
    }

    // Enables comparison of nodes owned by expressions. Considers the fields specified by the NodesCmpValues argument
    LIRComparator& enable(NodesCmpValues f) noexcept {
        m_nodes_cmp_values = static_cast<NodesCmpValues>(m_nodes_cmp_values | f);
        return *this;
    }

    // Enables comparison of expressions. Considers the fields specified by the LIRCmpValues argument
    LIRComparator& enable(LIRCmpValues f) noexcept {
        m_lir_cmp_values = static_cast<LIRCmpValues>(m_lir_cmp_values | f);
        return *this;
    }

    // Disables comparison of nodes owned by expressions. Considers the fields specified by the NodesCmpValues argument
    LIRComparator& disable(NodesCmpValues f) noexcept {
        m_nodes_cmp_values = static_cast<NodesCmpValues>(m_nodes_cmp_values & ~f);
        return *this;
    }

    // Disables comparison of expressions. Considers the fields specified by the LIRCmpValues argument
    LIRComparator& disable(LIRCmpValues f) noexcept {
        m_lir_cmp_values = static_cast<LIRCmpValues>(m_lir_cmp_values & ~f);
        return *this;
    }

    // Compares 2 Linear IRs based on enabled LIRCmpValues and NodesCmpValues
    Result compare(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                   const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir_ref);

private:
    explicit LIRComparator(NodesCmpValues nodes_vals, LIRCmpValues lir_vals) noexcept
        : m_nodes_cmp_values(nodes_vals),
          m_lir_cmp_values(lir_vals) {}

    bool should_compare(LIRCmpValues f) const noexcept {
        return m_lir_cmp_values & f;
    }

    static Result compare_descs(const std::vector<ov::snippets::lowered::PortDescriptorPtr>& descs,
                                const std::vector<ov::snippets::lowered::PortDescriptorPtr>& descs_ref);

    static Result compare_loop_managers(const ov::snippets::lowered::LoopManagerPtr& loop_manager,
                                        const ov::snippets::lowered::LoopManagerPtr& loop_manager_ref);

    static Result compare_loop_info(const ov::snippets::lowered::LoopInfoPtr& loop_info,
                                    const ov::snippets::lowered::LoopInfoPtr& loop_info_ref);

    static Result compare_unified_loop_info(const ov::snippets::lowered::UnifiedLoopInfoPtr& loop_info,
                                            const ov::snippets::lowered::UnifiedLoopInfoPtr& loop_info_ref);

    static Result compare_expaned_loop_info(const ov::snippets::lowered::ExpandedLoopInfoPtr& loop_info,
                                            const ov::snippets::lowered::ExpandedLoopInfoPtr& loop_info_ref);

    static Result compare_loop_ports(const std::vector<ov::snippets::lowered::LoopPort>& loop_ports,
                                     const std::vector<ov::snippets::lowered::LoopPort>& loop_ports_ref);

    static Result compare_expression_ports(const ov::snippets::lowered::ExpressionPort& expr_port,
                                           const ov::snippets::lowered::ExpressionPort& expr_port_ref);

    static Result compare_port_connectors(const std::vector<ov::snippets::lowered::PortConnectorPtr>& connectors,
                                          const std::vector<ov::snippets::lowered::PortConnectorPtr>& connectors_ref);

    static Result compare_handlers(const ov::snippets::lowered::SpecificIterationHandlers& handlers,
                                   const ov::snippets::lowered::SpecificIterationHandlers& handlers_ref);

    NodesCmpValues m_nodes_cmp_values;
    LIRCmpValues m_lir_cmp_values;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
