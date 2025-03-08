// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/define_buffer_clusters.hpp"

#include "snippets/lowered/pass/mark_invariant_shape_path.hpp"
#include "snippets/lowered/pass/set_buffer_reg_group.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {

// Find Loops which are connected to the current `buffer_expr` (consumer of Buffer is port of these Loops)
std::vector<size_t> get_connected_loops(const BufferExpressionPtr& buffer_expr, const ExpressionPtr& consumer_expr) {
    // [133463] Remove it please
    if (ov::is_type<op::LoopEnd>(consumer_expr->get_node()))
        return {};
    const auto& buffer_loops_ids = buffer_expr->get_loop_ids();
    const auto& consumer_loop_ids = consumer_expr->get_loop_ids();
    OPENVINO_ASSERT(buffer_loops_ids.size() <= consumer_loop_ids.size(), "Buffer with consumer are in incorrect loops");
    const auto mismatched_its = std::mismatch(buffer_loops_ids.begin(), buffer_loops_ids.end(), consumer_loop_ids.begin());
    return {mismatched_its.second, consumer_loop_ids.cend()};
}
} // namespace

using LoopPortInfo = UnifiedLoopInfo::LoopPortInfo;

DefineBufferClusters::BufferClusters::iterator DefineBufferClusters::find_cluster_by_expr(const BufferExpressionPtr& target) {
    return std::find_if(m_clusters.begin(), m_clusters.end(),
                        [&target](const BufferCluster& cluster) { return cluster.count(target) > 0; });
}

bool DefineBufferClusters::is_direct_buffer(const BufferExpressionPtr& buffer_expr, const ExpressionPtr& target_expr) {
    return buffer_expr && buffer_expr->get_loop_ids() == target_expr->get_loop_ids();
}

void DefineBufferClusters::create_new_cluster(const BufferExpressionPtr& buffer_expr) {
    const auto cluster_it = find_cluster_by_expr(buffer_expr);
    // If Buffer is missed in clusters, create new cluster with the single Buffer node inside
    if (cluster_it == m_clusters.cend()) {
        m_clusters.push_back(BufferCluster{buffer_expr});
    }
}

void DefineBufferClusters::add_buffers_to_cluster(BufferCluster& existing_cluster, const std::set<BufferExpressionPtr>& buffers) {
    existing_cluster.insert(buffers.cbegin(), buffers.cend());
    // All buffers in one cluster must be only static or dynamic (no mixes).
    if (std::any_of(existing_cluster.cbegin(), existing_cluster.cend(), [](const BufferExpressionPtr& buffer) { return !buffer->is_defined(); })) {
        for (const auto& buffer : existing_cluster)
            buffer->set_allocation_size(utils::get_dynamic_value<size_t>());
    }
}

size_t DefineBufferClusters::get_cluster_buffer_id(const BufferCluster& cluster) {
    OPENVINO_ASSERT(!cluster.empty(), "Buffer cluster is empty!");
    const auto id = cluster.cbegin()->get()->get_reg_group();
    if (std::all_of(cluster.cbegin(), cluster.cend(), [&id](const BufferExpressionPtr& expr) { return expr->get_reg_group() == id; })) {
        return id;
    }
    return SIZE_MAX;
}

std::pair<DefineBufferClusters::BufferMap, DefineBufferClusters::BufferMap> DefineBufferClusters::get_direct_buffers(const UnifiedLoopInfoPtr& loop_info,
                                                                                                                     const ExpressionPtr& loop_expr) {
    BufferMap input_buffers;
    const auto& loop_inputs = loop_info->get_input_ports_info();
    for (const auto& port_info : loop_inputs) {
        const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(port_info.port.get_expr_port()->get_port_connector_ptr()->get_source().get_expr());
        if (!is_direct_buffer(buffer_expr, loop_expr))
            continue;
        if (input_buffers.count(buffer_expr) > 0) {
            const auto& port_desc = port_info.desc;
            OPENVINO_ASSERT(input_buffers[buffer_expr].desc == port_desc,
                            "Invalid data pointer shifts: If Buffer has several consumers, this consumers must have the same shifts or zero");
            continue;
        }
        input_buffers[buffer_expr] = port_info;
    }

    BufferMap output_buffers;
    const auto& loop_outputs = loop_info->get_output_ports_info();
    for (const auto& port_info : loop_outputs) {
        const auto& consumer_inputs = port_info.port.get_expr_port()->get_port_connector_ptr()->get_consumers();
        for (const auto& consumer_input : consumer_inputs) {
            const auto& buffer_expr = ov::as_type_ptr<BufferExpression>(consumer_input.get_expr());
            if (!is_direct_buffer(buffer_expr, loop_expr))
                continue;
            OPENVINO_ASSERT(output_buffers.count(buffer_expr) == 0, "Only one Buffer can be on node output!");
            output_buffers[buffer_expr] = port_info;
        }
    }

    return std::make_pair(input_buffers, output_buffers);
}

void DefineBufferClusters::parse_loop(const LoopManagerPtr& loop_manager, const LinearIR::constExprIt& expr_it) {
    const auto& expr = *expr_it;
    const auto& loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
    const auto& loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_end->get_id());

    BufferMap input_buffers, output_buffers;
    std::tie(input_buffers, output_buffers) = get_direct_buffers(loop_info, expr);

    for (const auto& in : input_buffers)
        create_new_cluster(in.first);

    std::set<ExpressionPtr> visited_buffers;
    for (const auto& out : output_buffers) {
        const auto& output_buffer_expr = out.first;
        const auto& output_buffer_port_info = out.second;
        bool has_been_added = false;

        for (const auto& in : input_buffers) {
            const auto& input_buffer_expr = in.first;
            const auto& input_buffer_port_info = in.second;
            if (visited_buffers.count(input_buffer_expr) > 0)
                continue;

            // Memory can be reused if reading and writing are executed proportionally:
            //  - output buffer can have precision with data size less than input buffer
            if ((input_buffer_expr->get_data_type().size() < output_buffer_expr->get_data_type().size()))
                continue;

            const auto in_path = MarkInvariantShapePath::getInvariantPortShapePath(*input_buffer_port_info.port.get_expr_port());
            const auto out_path = MarkInvariantShapePath::getInvariantPortShapePath(*output_buffer_port_info.port.get_expr_port());
            //  - Memory can be reused if there are the same loop pointer increments (data size, final offsets, ptr increments).
            //    For that, loop ports with buffers should be on the same shape-path and have the same value of `is_incremented`.
            const auto in_is_incremented = input_buffer_port_info.port.is_incremented();
            const auto out_is_incremented = output_buffer_port_info.port.is_incremented();
            if (in_path != out_path || in_is_incremented != out_is_incremented)
                continue;

            //  - Memory can be shared if Buffer has the same allocation size.
            if (input_buffer_expr->is_defined() && output_buffer_expr->is_defined()) {
                if (input_buffer_expr->get_allocation_size() != output_buffer_expr->get_allocation_size())
                    continue;
            } else {
                // If allocation sizes are undefined, we can check if they have the same allocation sizes in runtime:
                //  - they should calculate allocation size using the common algorithm from `BufferExpression::init_allocation_size`.
                if (!utils::everyone_is(BufferExpression::get_type_info_static(), input_buffer_expr->get_type_info(), output_buffer_expr->get_type_info()))
                    continue;
            }

            const auto cluster_it = find_cluster_by_expr(input_buffer_expr);
            OPENVINO_ASSERT(cluster_it != m_clusters.end(), "Buffer on inputs of Loop must be already saved in clusters");
            // Add to the existing cluster
            add_buffers_to_cluster(*cluster_it, {output_buffer_expr});
            // Remove input buffer because we have already use its memory
            visited_buffers.insert(input_buffer_expr);
            has_been_added = true;
            break;
        }
        if (!has_been_added) {
            create_new_cluster(output_buffer_expr);
        }
    }

    // Check Buffers inside to possible memory reusing using `window` sliding
    parse_nested_loops(loop_manager, input_buffers, output_buffers, expr_it);
}

void DefineBufferClusters::parse_nested_loops(const LoopManagerPtr& loop_manager, const BufferMap& input_buffers,
                                              const BufferMap& output_buffers, const LinearIR::constExprIt& outer_loop_end_expr_it) {
    if (input_buffers.empty() && output_buffers.empty())
        return;

    auto can_be_data_ptr_proportionally_shifted = [](const LoopPortInfo& outer_port_info, const LoopPortInfo& inner_port_info) {
        // Outer Buffer ptr should be shifted to emulate "window" sliding
        const auto& outer_desc = outer_port_info.desc;
        if (!outer_port_info.port.is_incremented() || (!utils::is_dynamic_value(outer_desc.ptr_increment) && outer_desc.ptr_increment == 0))
            return false;

        OPENVINO_ASSERT(inner_port_info.port.get_expr_port() && outer_port_info.port.get_expr_port(), "Expression ports are nullptr!");
        // we can be sure that these data pointers will be proportionally shifted if they're on the same invariant shape path
        return MarkInvariantShapePath::getInvariantPortShapePath(*inner_port_info.port.get_expr_port()) ==
               MarkInvariantShapePath::getInvariantPortShapePath(*outer_port_info.port.get_expr_port());
    };

    const auto outer_loop_begin = ov::as_type_ptr<op::LoopEnd>(outer_loop_end_expr_it->get()->get_node())->get_loop_begin();
    for (auto it = std::reverse_iterator<LinearIR::constExprIt>(outer_loop_end_expr_it); (*it)->get_node() != outer_loop_begin; ++it) {
        const auto& inner_expr = *it;
        if (const auto inner_buffer_expr = ov::as_type_ptr<BufferExpression>(inner_expr)) {
            const auto inner_cluster_it = find_cluster_by_expr(inner_buffer_expr);
            OPENVINO_ASSERT(inner_cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
            const auto inner_cluster_id = get_cluster_buffer_id(*inner_cluster_it);
            if (inner_cluster_id == SIZE_MAX) continue;

            // If inner Buffer is not connected to the Loop - `window` sliding effect is not possible
            LoopPortInfo final_loop_info;
            if (!init_buffer_last_loop_port_info(loop_manager, inner_buffer_expr, final_loop_info))
                continue;

            auto unite = [&](const BufferMap& ports, const bool is_input) {
                bool applied = false;
                for (const auto& port : ports) {
                    const auto cluster_it = find_cluster_by_expr(port.first);
                    OPENVINO_ASSERT(cluster_it != m_clusters.cend(), "Buffer cluster has not been found");
                    // If the buffers are already in the same cluster or have different Buffer ID - skip
                    if (cluster_it == inner_cluster_it) continue;
                    // Buffer from one cluster must be only defined (with known allocation_size) or dynamic (with unknown allocation_size)
                    if (inner_buffer_expr->is_defined() != port.first->is_defined()) continue;
                    // The inner Buffer can reuse memory of the outer Buffer using `window` sliding only if:
                    //  - The finalization offset of the latest Loop connected to the inner Buffer is equal to
                    //    pointer increment of outer Buffer to emulate `window` sliding
                    //  - This outer Buffer should have the same Buffer ID as inner to move data ptr of inner Buffer after each outer Loop iteration.
                    //    It's needed because all Loops reset data pointers of connected Buffer after full work.
                    //    To avoid rewriting of outer Buffer data we have to have the same Buffer ID (GPR) to proportionally shift pointers both Buffers.
                    if (!can_be_data_ptr_proportionally_shifted(port.second, final_loop_info)) continue;

                    applied = unite_nested_clusters(loop_manager, inner_cluster_it, *cluster_it, port.first, is_input);
                    if (applied) break;
                }
                return applied;
            };

            if (unite(input_buffers, true)) continue;
            if (unite(output_buffers, false)) continue;
        }
    }
}

bool DefineBufferClusters::init_buffer_last_loop_port_info(const LoopManagerPtr& loop_manager, const BufferExpressionPtr& buffer_expr,
                                                           UnifiedLoopInfo::LoopPortInfo& port_info) {
    auto get_direct_loop_for_buffer_out = [&](const BufferExpressionPtr& buffer_expr, const ExpressionPtr& consumer_expr) -> UnifiedLoopInfoPtr {
        const auto inner_loops = get_connected_loops(buffer_expr, consumer_expr);
        if (inner_loops.empty())
            return nullptr;
        return loop_manager->get_loop_info<UnifiedLoopInfo>(inner_loops.front());
    };

    bool found = false;
    double last_loop_exec_order = -1 * std::numeric_limits<double>::max();
    const auto& buffer_outs = buffer_expr->get_output_port_connectors();
    for (const auto& buffer_out : buffer_outs) {
        const auto consumers = buffer_out->get_consumers();
        for (const auto& consumer : consumers) {
            if (const auto& direct_loop = get_direct_loop_for_buffer_out(buffer_expr, consumer.get_expr())) {
                const auto loop_order = direct_loop->get_output_ports().back().get_expr_port()->get_expr()->get_exec_num();
                if (loop_order > last_loop_exec_order) {
                    OPENVINO_ASSERT(direct_loop->is_loop_port(consumer), "Consumer of Buffer from another loop must be loop port");
                    port_info = direct_loop->get_loop_port_info(consumer);
                    last_loop_exec_order = loop_order;
                    found = true;
                }
            }
        }
    }
    return found;
}

bool DefineBufferClusters::unite_nested_clusters(const LoopManagerPtr& loop_manager, const BufferClusters::iterator& inner_cluster_it,
                                                 BufferCluster& outer_cluster, const BufferExpressionPtr& outer_buffer, bool is_outer_up) {
    for (const auto& inner_buffer : *inner_cluster_it) {
        const auto& upper_buffer = is_outer_up ? outer_buffer : inner_buffer;
        const auto& lower_buffer = is_outer_up ? inner_buffer : outer_buffer;

        const auto& lower_buffer_source = lower_buffer->get_input_port_connector(0)->get_source();
        const auto& upper_buffer_consumers = upper_buffer->get_output_port_connector(0)->get_consumers();
        for (const auto& upper_buffer_consumer : upper_buffer_consumers) {
            const auto& connected_loops = get_connected_loops(upper_buffer, upper_buffer_consumer.get_expr());
            for (const auto& loop_id : connected_loops) {
                const auto& common_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
                if (!common_loop_info->is_loop_port(lower_buffer_source) || !common_loop_info->is_loop_port(upper_buffer_consumer))
                    continue;

                const auto upper_port_desc = common_loop_info->get_loop_port_info(upper_buffer_consumer);
                const auto lower_port_desc = common_loop_info->get_loop_port_info(lower_buffer_source);
                if (SetBufferRegGroup::can_be_in_one_reg_group(upper_port_desc, lower_port_desc)) {
                    for (const auto& inner_buffer : *inner_cluster_it)
                        inner_buffer->set_reg_group(outer_buffer->get_reg_group());

                    add_buffers_to_cluster(outer_cluster, *inner_cluster_it);
                    m_clusters.erase(inner_cluster_it);
                    return true;
                }
            }
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
            parse_loop(linear_ir.get_loop_manager(), expr_it);
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
