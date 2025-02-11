// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_buffers.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool ValidateBuffers::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateBuffers")

    const auto& lir_buffers = linear_ir.get_buffers();

    // Firstly we check that all BufferExpression are in "get_buffers()"
    for (const auto& expr : linear_ir) {
        if (const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(expr))
            OPENVINO_ASSERT(std::find(lir_buffers.cbegin(), lir_buffers.cend(), buffer_expr) != lir_buffers.cend(),
                            "All BufferExpressions must be in LinearIR.get_buffers()");
    }

    // Secondly we should validate buffers and their clusters
    std::set<size_t> cluster_ids;
    std::map<size_t, std::set<lowered::BufferExpressionPtr>> dynamic_buffer_clusters, static_buffer_clusters;
    for (const auto& buffer_expr : lir_buffers) {
        // TODO [143395] : MemoryManager should provide exact containers with needed buffers (static or dynamic) without any `is_defined()`
        auto& clusters = buffer_expr->is_defined() ? static_buffer_clusters : dynamic_buffer_clusters;
        clusters[buffer_expr->get_cluster_id()].insert(buffer_expr);
        cluster_ids.insert(buffer_expr->get_cluster_id());

        buffer_expr->validate();
    }

    OPENVINO_ASSERT(cluster_ids.size() == dynamic_buffer_clusters.size() + static_buffer_clusters.size(), "Incorrect count of Buffer clusters");
    OPENVINO_ASSERT(cluster_ids.empty() || (*cluster_ids.cbegin() == 0 && *cluster_ids.crbegin() == (cluster_ids.size() - 1)),
                    "Incorrect indetifiers of Buffer clusters");

    for (const auto& p : static_buffer_clusters) {
        const auto& cluster_id = p.first;
        const auto& cluster = p.second;
        OPENVINO_ASSERT(dynamic_buffer_clusters.count(cluster_id) == 0, "Buffers from the same cluster must be only static or dynamic");

        OPENVINO_ASSERT(cluster.size() > 0, "Incorrect size of buffer cluster");
        size_t cluster_offset = (*cluster.cbegin())->get_offset();
        for (const auto& buffer_expr : cluster) {
            OPENVINO_ASSERT(cluster_offset == buffer_expr->get_offset(), "Static Buffers from the same cluster must have the same offset!");
        }
    }

    return !lir_buffers.empty();
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
