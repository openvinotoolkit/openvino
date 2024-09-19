// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/define_buffer_clusters.hpp"

#include "snippets/lowered/pass/set_buffer_reg_group.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using ShiftPtrParams = SetBufferRegGroup::ShiftPtrParams;

DefineBufferClusters::BufferClusters::iterator DefineBufferClusters::find_cluster_by_expr(const BufferExpressionPtr& target) {
    return std::find_if(m_clusters.begin(), m_clusters.end(),
                        [&target](const BufferCluster& cluster) { return cluster.count(target) > 0; });
}

bool DefineBufferClusters::is_direct_buffer(const BufferExpressionPtr& buffer_expr, const ExpressionPtr& target_expr) const {
    return buffer_expr && buffer_expr->get_loop_ids() == target_expr->get_loop_ids();
}

void DefineBufferClusters::create_new_cluster(const BufferExpressionPtr& buffer_expr) {
    const auto cluster_it = find_cluster_by_expr(buffer_expr);
    // If Buffer is missed in clusters, create new cluster with the single Buffer node inside
    if (cluster_it == m_clusters.cend()) {
        m_clusters.push_back(BufferCluster{buffer_expr});
    }
}

size_t DefineBufferClusters::get_cluster_buffer_id(const BufferCluster& cluster) const {
    OPENVINO_ASSERT(!cluster.empty(), "Buffer cluster is empty!");
    const auto id = cluster.cbegin()->get()->get_reg_group();
    if (std::all_of(cluster.cbegin(), cluster.cend(), [&id](const BufferExpressionPtr& expr) { return expr->get_reg_group() == id; })) {
        return id;
    }
    return SIZE_MAX;
}

DefineBufferClusters::BufferPorts DefineBufferClusters::get_input_buffers(const ExpressionPtr& loop_expr) const {
    BufferPorts input_buffers;

    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(loop_expr->get_node());
    const auto in_count = loop_end->get_input_num();
    const auto& connectors = loop_expr->get_input_port_connectors();

    // Input Buffers
    for (size_t i = 0; i < in_count; ++i) {
        const auto& source_expr = ov::as_type_ptr<BufferExpression>(connectors[i]->get_source().get_expr());
        if (!is_direct_buffer(source_expr, loop_expr))
            continue;
        // Save as input Buffer
        const auto ret = input_buffers.insert(std::make_pair(source_expr, std::set<size_t>{ i })).second;
        if (!ret)
            input_buffers[source_expr].insert(i);
    }
    return input_buffers;
}

DefineBufferClusters::BufferPorts DefineBufferClusters::get_output_buffers(const ExpressionPtr& loop_expr) const {
    BufferPorts output_buffers;

    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(loop_expr->get_node());
    const auto in_count = loop_end->get_input_num();
    const auto out_count = loop_end->get_output_num();
    const auto& connectors = loop_expr->get_input_port_connectors();

    for (size_t i = in_count; i < in_count + out_count; ++i) {
        for (const auto& consumer : connectors[i]->get_consumers()) {
            const auto& consumer_expr =  ov::as_type_ptr<BufferExpression>(consumer.get_expr());
            if (!is_direct_buffer(consumer_expr, loop_expr))
                continue;
            // Save as output Buffer
            output_buffers[consumer_expr] = { i };
        }
    }
    return output_buffers;
}

void DefineBufferClusters::parse_loop(const LinearIR::constExprIt& expr_it) {
    const auto& expr = *expr_it;
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& final_offsets = loop_end->get_finalization_offsets();
    const auto& data_sizes = loop_end->get_element_type_sizes();

    // [ Expression -> Port indexes ]
    const auto input_buffers = get_input_buffers(expr);
    const auto output_buffers = get_output_buffers(expr);

    for (const auto& in : input_buffers)
        create_new_cluster(in.first);

    std::set<ExpressionPtr> visited_buffers;
    for (const auto& out : output_buffers) {
        const auto output_buffer_expr = out.first;
        const auto output_buffer_port_idx = *(out.second.cbegin());  // Output port is always one
        bool has_been_added = false;

        for (const auto& in : input_buffers) {
            const auto& input_buffer_expr = in.first;
            if (visited_buffers.count(input_buffer_expr) > 0)
                continue;

            // If allocated sizes of buffers are unkown on compilation stage (dynamic),
            // we cannot be sure that they're will be the same in runtime.
            if (!input_buffer_expr->is_defined()|| !output_buffer_expr->is_defined())
                continue;

            // Memory can be reused if reading and writing are executed proportionally:
            //  - the same reading/writing order
            //  - the same buffer memory sizes
            if ((input_buffer_expr->get_byte_size() != output_buffer_expr->get_byte_size()) ||
                (input_buffer_expr->get_output_port_descriptor(0)->get_layout() != output_buffer_expr->get_input_port_descriptor(0)->get_layout()))
                continue;

            // Also memory can be reused if there are the same ShiftPtrParams (data size, final offsets, ptr increments)
            const auto& input_buffer_ports = in.second;
            for (const auto& input_buffer_port_idx : input_buffer_ports) {
                const auto input_params =
                    ShiftPtrParams(data_sizes[input_buffer_port_idx], ptr_increments[input_buffer_port_idx], final_offsets[input_buffer_port_idx]);
                const auto output_params =
                    ShiftPtrParams(data_sizes[output_buffer_port_idx], ptr_increments[output_buffer_port_idx], final_offsets[output_buffer_port_idx]);

                // If data pointer shift parameters are unknown on model compilation stage (dynamic),
                // we cannot be sure that these data pointers will be proportionally shifted in runtime.
                if (input_params.is_static() && output_params.is_static() && input_params == output_params) {
                    const auto cluster_it = find_cluster_by_expr(input_buffer_expr);
                    OPENVINO_ASSERT(cluster_it != m_clusters.end(), "Buffer on inputs of Loop must be already saved in clusters");
                    // Add to the existing cluster
                    has_been_added = cluster_it->insert(output_buffer_expr).second;
                    OPENVINO_ASSERT(has_been_added, "Buffer has not been saved in cluster");
                    // Remove input buffer because we have already use its memory
                    visited_buffers.insert(input_buffer_expr);
                    break;
                }
            }
            if (has_been_added) break;
        }
        if (!has_been_added) {
            m_clusters.push_back(BufferCluster{output_buffer_expr});
        }
    }

    // Check Buffers inside to possible memory reusing using `window` sliding
    parse_nested_loops(input_buffers, output_buffers, expr_it);
}

void DefineBufferClusters::parse_nested_loops(const BufferPorts& input_buffers, const BufferPorts& output_buffers,
                                              const LinearIR::constExprIt& outer_loop_end_expr_it) {
    if (input_buffers.empty() && output_buffers.empty())
        return;

    // The inner Buffer can reuse memory of the outer Buffer using `window` sliding only if:
    //  - The finalization offset of the latest Loop connected to the inner Buffer is equal to pointer increment of outer Buffer to emulate `window` sliding
    //  - This outer Buffer should have the same Buffer ID as inner to move data ptr of inner Buffer after each outer Loop iteration.
    //    It's needed because all Loops reset data pointers of connected Buffer after full work.
    //    To avoid rewriting of outer Buffer data we have to have the same Buffer ID (GPR) to proportionally shift pointers both Buffers.

    auto can_be_data_ptr_proportionally_shifted = [](int64_t outer_buffer_ptr_increment, int64_t outer_buffer_data_size,
                                                     int64_t inner_buffer_final_offsets, int64_t inner_buffer_data_size) {
        // If data pointer shift parameters are unknown on model compilation stage (dynamic),
        // we cannot be sure that these data pointers will be proportionally shifted in runtime.
        if (utils::is_dynamic_value(outer_buffer_ptr_increment) || utils::is_dynamic_value(inner_buffer_final_offsets))
            return false;
        return (outer_buffer_ptr_increment != 0) &&
               ((inner_buffer_data_size * inner_buffer_final_offsets * -1) == outer_buffer_ptr_increment * outer_buffer_data_size);
    };

    const auto outer_loop_end = ov::as_type_ptr<op::LoopEnd>(outer_loop_end_expr_it->get()->get_node());
    const auto outer_loop_begin = outer_loop_end->get_loop_begin();
    const auto& outer_ptr_increments = outer_loop_end->get_ptr_increments();
    const auto& outer_data_sizes = outer_loop_end->get_element_type_sizes();

    for (auto it = std::reverse_iterator<LinearIR::constExprIt>(outer_loop_end_expr_it); (*it)->get_node() != outer_loop_begin; ++it) {
        const auto& inner_expr = *it;
        if (const auto inner_buffer_expr = ov::as_type_ptr<BufferExpression>(inner_expr)) {
            const auto inner_cluster_it = find_cluster_by_expr(inner_buffer_expr);
            OPENVINO_ASSERT(inner_cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
            const auto inner_cluster_id = get_cluster_buffer_id(*inner_cluster_it);
            if (inner_cluster_id == SIZE_MAX) continue;

            const auto final_offset = get_buffer_finalization_offset(inner_buffer_expr);

            auto unite = [&](const BufferPorts& ports, const bool is_input) {
                bool applied = false;
                for (const auto& port : ports) {
                    const auto cluster_it = find_cluster_by_expr(port.first);
                    OPENVINO_ASSERT(cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
                    // If the buffers are already in the same cluster or have different Buffer ID - skip
                    if (cluster_it == inner_cluster_it) continue;
                    // Buffer from one cluster must be only defined (with known allocation_size) or dynamic (with unknown allocation_size)
                    if (inner_buffer_expr->is_defined() != port.first->is_defined()) continue;

                    bool can_be_reused = true;
                    for (const auto idx : port.second) {
                        can_be_reused = can_be_reused &&
                            can_be_data_ptr_proportionally_shifted(outer_ptr_increments[idx], outer_data_sizes[idx],
                                                                   final_offset, inner_buffer_expr->get_node()->get_element_type().size());
                    }
                    if (!can_be_reused)
                        continue;

                    applied = unite_nested_clusters(inner_cluster_it, *cluster_it, port.first, is_input);
                    if (applied) break;
                }
                return applied;
            };

            if (unite(input_buffers, true)) continue;
            if (unite(output_buffers, false)) continue;
        }
    }
}

int64_t DefineBufferClusters::get_buffer_finalization_offset(const BufferExpressionPtr& buffer_expr) const {
    auto index = [](const std::vector<PortConnectorPtr>& loop_inputs, const PortConnectorPtr& buffer_out) {
        const auto it = std::find(loop_inputs.cbegin(), loop_inputs.cend(), buffer_out);
        OPENVINO_ASSERT(it != loop_inputs.cend(), "Buffer output PortConnector has not been found in target LoopEnd inputs");
        return std::distance(loop_inputs.cbegin(), it);
    };
    int64_t final_offset = 0;
    double last_loop_exec_order = -1 * std::numeric_limits<double>::max();
    const auto& buffer_outs = buffer_expr->get_output_port_connectors();
    for (const auto& buffer_out : buffer_outs) {
        const auto consumers = buffer_out->get_consumers();
        for (const auto& consumer : consumers) {
            const auto consumer_expr = consumer.get_expr();
            const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(consumer_expr->get_node());
            if (loop_end && consumer_expr->get_loop_ids() == buffer_expr->get_loop_ids()) {
                const auto loop_order = consumer_expr->get_exec_num();
                if (loop_order > last_loop_exec_order) {
                    const auto& loop_inputs = consumer_expr->get_input_port_connectors();
                    final_offset = loop_end->get_finalization_offsets()[index(loop_inputs, buffer_out)];
                    last_loop_exec_order = loop_order;
                }
            }
        }
    }
    return final_offset;
}

bool DefineBufferClusters::unite_nested_clusters(const BufferClusters::iterator& inner_cluster_it,
                                                 BufferCluster& outer_cluster,
                                                 const BufferExpressionPtr& outer_buffer, bool is_outer_up) {
    for (const auto& inner_buffer : *inner_cluster_it) {
        ExpressionPtr common_loop_end_expr = nullptr;
        size_t outer_idx = SIZE_MAX, inner_idx = SIZE_MAX;
        const auto& up_buffer = is_outer_up ? outer_buffer : inner_buffer;
        const auto& down_buffer = is_outer_up ? inner_buffer : outer_buffer;
        auto& up_idx = is_outer_up ? outer_idx : inner_idx;
        auto& down_idx = is_outer_up ? inner_idx : outer_idx;
        if (are_buffer_neighbours(up_buffer, down_buffer, common_loop_end_expr, up_idx, down_idx)) {
            const auto common_loop_end = ov::as_type_ptr<op::LoopEnd>(common_loop_end_expr->get_node());
            const auto& inner_ptr_increments = common_loop_end->get_ptr_increments();
            const auto& inner_final_offsets = common_loop_end->get_finalization_offsets();
            const auto& inner_data_sizes = common_loop_end->get_element_type_sizes();
            if (SetBufferRegGroup::can_be_in_one_group({ inner_data_sizes[up_idx], inner_ptr_increments[up_idx], inner_final_offsets[up_idx] },
                                                       { inner_data_sizes[down_idx], inner_ptr_increments[down_idx], inner_final_offsets[down_idx] })) {
                for (const auto& inner_buffer : *inner_cluster_it)
                    inner_buffer->set_reg_group(outer_buffer->get_reg_group());

                outer_cluster.insert(inner_cluster_it->cbegin(), inner_cluster_it->cend());
                m_clusters.erase(inner_cluster_it);
                return true;
            }
        }
    }
    return false;
}

bool DefineBufferClusters::are_buffer_neighbours(const BufferExpressionPtr& up, const BufferExpressionPtr& down, ExpressionPtr& loop,
                                                 size_t& up_idx, size_t& down_idx) {
    auto find_input = [&down](const PortConnectorPtr& in) {
        return in->get_source().get_expr() == down;
    };
    auto find_output = [&down](const PortConnectorPtr& in) {
        const auto consumers = in->get_consumers();
        return std::any_of(consumers.cbegin(), consumers.cend(),
                           [&down](const ExpressionPort& port) { return port.get_expr() == down; });
    };
    auto find = [&](const std::vector<PortConnectorPtr>::const_iterator& begin,
                    const std::vector<PortConnectorPtr>::const_iterator& end,
                    const std::vector<PortConnectorPtr>::const_iterator& orig_begin,
                    const ExpressionPort& loop_port,
                    bool is_input) -> bool {
        const auto in_buffer_it = is_input ? std::find_if(begin, end, find_input)
                                           : std::find_if(begin, end, find_output);
        if (in_buffer_it != end) {
            up_idx = loop_port.get_index();
            down_idx = std::distance(orig_begin, in_buffer_it);
            loop = loop_port.get_expr();
            return true;
        }
        return false;
    };
    for (const auto& out : up->get_output_port_connectors()) {
        for (const auto& buffer_consumer : out->get_consumers()) {
            const auto buffer_consumer_expr = buffer_consumer.get_expr();
            const auto loop_end = ov::as_type_ptr<op::LoopEnd>(buffer_consumer_expr->get_node());
            if (!loop_end)
                continue;
            const auto& loop_inputs = buffer_consumer_expr->get_input_port_connectors();
            if (find(loop_inputs.cbegin(), loop_inputs.cbegin() + loop_end->get_input_num(), loop_inputs.cbegin(), buffer_consumer, true)) return true;
            if (find(loop_inputs.cbegin() + loop_end->get_input_num(), loop_inputs.cend(), loop_inputs.cbegin(), buffer_consumer, false)) return true;
        }
    }
    return false;
}

void DefineBufferClusters::parse_memory_access_op(const ExpressionPtr& expr) {
    const auto ma = std::dynamic_pointer_cast<modifier::MemoryAccess>(expr->get_node());
    // TODO: Some full MemoryAccess ops can have inplace inputs and outputs in general.
    //       Need to add mechanism of inplace ports using MemoryAccess::PortDescriptor::inplace
    for (const auto& input : expr->get_input_port_connectors()) {
        const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(input->get_source().get_expr());
        if (is_direct_buffer(buffer_expr, expr))
            create_new_cluster(buffer_expr);
    }
    for (const auto& output : expr->get_output_port_connectors()) {
        for (const auto& consumer : output->get_consumers()) {
            const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(consumer.get_expr());
            if (is_direct_buffer(buffer_expr, expr))
                create_new_cluster(buffer_expr);
        }
    }
}

bool DefineBufferClusters::run(lowered::LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::DefineBufferClusters");

    m_clusters.clear();

    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto op = expr->get_node();
        if (ov::is_type<op::LoopEnd>(op)) {
            parse_loop(expr_it);
            continue;
        }

        if (std::dynamic_pointer_cast<modifier::MemoryAccess>(op)) {
            parse_memory_access_op(expr);
            continue;
        }
    }

    for (size_t cluster_id = 0; cluster_id < m_clusters.size(); ++cluster_id) {
        const auto& cluster = m_clusters[cluster_id];
        std::for_each(cluster.cbegin(), cluster.cend(), [&cluster_id](const BufferExpressionPtr& buffer_expr) {
            buffer_expr->set_cluster_id(cluster_id);
        });
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
