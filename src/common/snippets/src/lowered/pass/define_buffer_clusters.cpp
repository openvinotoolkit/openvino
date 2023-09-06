// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/define_buffer_clusters.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using BufferCluster = AllocateBufferMemory::BufferCluster;
using BufferClusters = AllocateBufferMemory::BufferClusters;

BufferClusters::iterator DefineBufferClusters::find_cluster_by_expr(const ExpressionPtr& target) {
    return std::find_if(m_clusters.begin(), m_clusters.end(),
                        [&target](const AllocateBufferMemory::BufferCluster& cluster) { return cluster.count(target) > 0; });
}

bool DefineBufferClusters::is_direct_buffer(const ExpressionPtr& buffer_expr, const ExpressionPtr& target_expr) const {
    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
    return buffer && buffer_expr->get_loop_ids() == target_expr->get_loop_ids();
}

void DefineBufferClusters::create_new_cluster(const ExpressionPtr& buffer_expr) {
    const auto cluster_it = find_cluster_by_expr(buffer_expr);
    // If Buffer is missed in clusters, create new cluster with the single Buffer node inside
    if (cluster_it == m_clusters.cend()) {
        m_clusters.push_back(BufferCluster{buffer_expr});
    }
}

size_t DefineBufferClusters::get_cluster_buffer_id(const AllocateBufferMemory::BufferCluster& cluster) const {
    OPENVINO_ASSERT(!cluster.empty(), "Buffer cluster is empty!");
    const auto id = (ov::as_type_ptr<op::Buffer>(cluster.cbegin()->get()->get_node()))->get_id();
    if (std::all_of(cluster.cbegin(), cluster.cend(),
                    [&id](const ExpressionPtr& expr) { return (ov::as_type_ptr<op::Buffer>(expr->get_node()))->get_id() == id; })) {
        return id;
    }
    return SIZE_MAX;
}

void DefineBufferClusters::parse_loop(const LinearIR::constExprIt& expr_it) {
    const auto& expr = *expr_it;
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
    const auto ptr_increments = loop_end->get_ptr_increments();
    const auto final_offsets = loop_end->get_finalization_offsets();
    const auto in_count = loop_end->get_input_num();
    const auto out_count = loop_end->get_output_num();
    const auto connectors = expr->get_input_port_connectors();

    std::set<ExpressionPtr> visited_buffers;
    // [ Expression -> Port indexes ]
    std::unordered_map<ExpressionPtr, std::set<size_t>> input_buffers;
    std::unordered_map<ExpressionPtr, size_t> output_buffers;
    for (size_t i = 0; i < in_count; ++i) {
        const auto source_expr = connectors[i]->get_source().get_expr();
        if (!is_direct_buffer(source_expr, expr))
            continue;

        create_new_cluster(source_expr);
        // Save as input Buffer
        const auto ret = input_buffers.insert(std::make_pair(source_expr, std::set<size_t>{ i })).second;
        if (!ret)
            input_buffers[source_expr].insert(i);
    }

    for (size_t i = in_count; i < in_count + out_count; ++i) {
        for (const auto& consumer : connectors[i]->get_consumers()) {
            auto consumer_expr = consumer.get_expr();
            if (!is_direct_buffer(consumer_expr, expr))
                continue;

            const auto buffer = ov::as_type_ptr<op::Buffer>(consumer_expr->get_node());
            bool has_been_added = false;
            for (const auto& input_buffer : input_buffers) {
                const auto& input_buffer_expr = input_buffer.first;
                if (visited_buffers.count(input_buffer_expr) > 0)
                    continue;
                const auto input_buffer_node = ov::as_type_ptr<op::Buffer>(input_buffer_expr->get_node());
                const auto& input_buffer_idxs = input_buffer.second;
                for (const auto& input_buffer_idx : input_buffer_idxs) {
                    if (input_buffer_node->get_byte_size() == buffer->get_byte_size() &&
                        input_buffer_expr->get_output_port_descriptor(0)->get_layout() == consumer.get_descriptor_ptr()->get_layout() &&
                        ptr_increments[input_buffer_idx] == ptr_increments[i] &&
                        final_offsets[input_buffer_idx] == final_offsets[i]) {
                        const auto cluster_it = find_cluster_by_expr(input_buffer_expr);
                        OPENVINO_ASSERT(cluster_it != m_clusters.end(), "Buffer on inputs of Loop must be already saved in clusters");
                        // Add to the existing cluster
                        has_been_added = cluster_it->insert(consumer_expr).second;
                        OPENVINO_ASSERT(has_been_added, "Buffer has not been saved in cluster");
                        // Remove input buffer because we have already use its memory
                        visited_buffers.insert(input_buffer_expr);
                        break;
                    }
                }
                if (has_been_added) break;
            }
            if (!has_been_added) {
                m_clusters.push_back(BufferCluster{consumer_expr});
            }
            output_buffers[consumer_expr] = i;
        }
    }

    unite_clusters_in_nested_loops(input_buffers, output_buffers, expr_it);
}

void DefineBufferClusters::unite_clusters_in_nested_loops(const std::unordered_map<ExpressionPtr, std::set<size_t>>& input_buffers,
                                                          const std::unordered_map<ExpressionPtr, size_t>& output_buffers,
                                                          const LinearIR::constExprIt& outer_loop_end_expr_it) {
    if (input_buffers.empty() && output_buffers.empty())
        return;

    auto cluster_contains_buffer_id = [](const BufferCluster& cluster,  size_t target_buffer_id) {
        auto the_same_buffer_id = [target_buffer_id](const ExpressionPtr& expr) {
            return (ov::as_type_ptr<op::Buffer>(expr->get_node()))->get_id() == target_buffer_id;
        };
        return std::any_of(cluster.cbegin(), cluster.cend(), the_same_buffer_id);
    };

    auto can_be_data_ptr_proportionally_shifted = [](int64_t outer_buffer_ptr_increment, int64_t outer_buffer_data_size,
                                                     int64_t inner_buffer_final_offsets, int64_t inner_buffer_data_size) {
        return (outer_buffer_ptr_increment != 0) &&
               ((inner_buffer_data_size * inner_buffer_final_offsets * -1) == outer_buffer_ptr_increment * outer_buffer_data_size);
    };

    const auto outer_loop_end = ov::as_type_ptr<op::LoopEnd>(outer_loop_end_expr_it->get()->get_node());
    const auto outer_loop_begin = outer_loop_end->get_loop_begin();
    for (auto it = std::reverse_iterator<LinearIR::constExprIt>(outer_loop_end_expr_it); (*it)->get_node() != outer_loop_begin; ++it) {
        const auto inner_expr = *it;
        if (const auto inner_buffer = ov::as_type_ptr<op::Buffer>(inner_expr->get_node())) {
            auto inner_cluster_it = find_cluster_by_expr(inner_expr);
            OPENVINO_ASSERT(inner_cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
            const auto inner_cluster_id = get_cluster_buffer_id(*inner_cluster_it);
            if (inner_cluster_id == SIZE_MAX) continue;

            const auto final_offset = get_buffer_finalization_offsets(inner_expr);

            for (const auto& in : input_buffers) {
                const auto cluster_it = find_cluster_by_expr(in.first);
                OPENVINO_ASSERT(cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
                if (cluster_it == inner_cluster_it || !cluster_contains_buffer_id(*inner_cluster_it, inner_cluster_id)) continue;

                bool can_be_reused = true;
                for (const auto idx : in.second) {
                    can_be_reused = can_be_reused &&
                        can_be_data_ptr_proportionally_shifted(outer_loop_end->get_ptr_increments()[idx], outer_loop_end->get_element_type_sizes()[idx],
                                                               final_offset, inner_buffer->get_element_type().size());
                }
                if (!can_be_reused)
                    continue;

                cluster_it->insert(inner_cluster_it->cbegin(), inner_cluster_it->cend());
                m_clusters.erase(inner_cluster_it);
                inner_cluster_it = m_clusters.end();
                break;
            }
            if (inner_cluster_it == m_clusters.end()) continue;

            for (const auto& out : output_buffers) {
                const auto cluster_it = find_cluster_by_expr(out.first);
                OPENVINO_ASSERT(cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
                if (cluster_it == inner_cluster_it || !cluster_contains_buffer_id(*inner_cluster_it, inner_cluster_id)) continue;

                if (!can_be_data_ptr_proportionally_shifted(
                    outer_loop_end->get_ptr_increments()[out.second], outer_loop_end->get_element_type_sizes()[out.second],
                    final_offset, inner_buffer->get_element_type().size()))
                    continue;

                cluster_it->insert(inner_cluster_it->cbegin(), inner_cluster_it->cend());
                m_clusters.erase(inner_cluster_it);
                break;
            }
        }
    }
}

int64_t DefineBufferClusters::get_buffer_finalization_offsets(const ExpressionPtr& buffer_expr) const {
    auto index = [](const std::vector<PortConnectorPtr>& loop_inputs, const PortConnectorPtr& buffer_out) {
        const auto it = std::find(loop_inputs.cbegin(), loop_inputs.cend(), buffer_out);
        OPENVINO_ASSERT(it != loop_inputs.cend(), "Buffer output PortConnector has not been found in target LoopEnd inputs");
        return std::distance(loop_inputs.cbegin(), it);
    };
    int64_t final_offset;
    size_t last_loop_exec_order = 0;
    const auto buffer_outs = buffer_expr->get_output_port_connectors();
    for (const auto& buffer_out : buffer_outs) {
        const auto consumers = buffer_out->get_consumers();
        for (const auto& consumer : consumers) {
            const auto consumer_expr = consumer.get_expr();
            const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(consumer_expr->get_node());
            if (loop_end && consumer_expr->get_loop_ids() == buffer_expr->get_loop_ids()) {
                const auto loop_order = ov::snippets::pass::GetTopologicalOrder(loop_end);
                if (loop_order > last_loop_exec_order) {
                    const auto loop_inputs = consumer_expr->get_input_port_connectors();
                    final_offset = loop_end->get_finalization_offsets()[index(loop_inputs, buffer_out)];
                    last_loop_exec_order = loop_order;
                }
            }
        }
    }
    return final_offset;
}

void DefineBufferClusters::parse_memory_access_op(const ExpressionPtr& expr) {
    const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
    if (!ma->is_full_memory_access_op())
        return;
    for (const auto& input : expr->get_input_port_connectors()) {
        if (is_direct_buffer(input->get_source().get_expr(), expr)) {
            create_new_cluster(input->get_source().get_expr());
        }
    }
    for (const auto& output : expr->get_output_port_connectors()) {
        for (const auto& consumer : output->get_consumers()) {
            if (is_direct_buffer(consumer.get_expr(), expr)) {
                create_new_cluster(consumer.get_expr());
            }
        }
    }
}

bool DefineBufferClusters::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::DefineBufferClusters");

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto expr = *expr_it;
        const auto op = expr->get_node();
        if (ov::is_type<op::LoopEnd>(op)) {
            parse_loop(expr_it);
            continue;
        }

        // TODO: Some full MemoryAccess ops can have inplace inputs and outputs in general.
        //       Need to add mechanism of inplace ports using MemoryAccess::PortDescriptor::inplace
        if (ov::is_type<op::MemoryAccess>(op)) {
            parse_memory_access_op(expr);
            continue;
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
