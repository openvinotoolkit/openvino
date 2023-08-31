// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "snippets/lowered/buffer_solver.hpp"

#include "snippets/lowered/pass/identify_buffers.hpp"
#include "snippets/lowered/pass/clean_repeated_ptr_shifts.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {

int64_t BufferSolver::solve(lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BufferSolver::solve")

    enumerate(linear_ir);
    identify_buffers(linear_ir);
    const auto buffer_clusters = init_clusters(linear_ir);
    const auto scratchpad_size = allocate(buffer_clusters);
    lowered::pass::CleanRepeatedDataPointerShifts pass;
    pass.run(linear_ir);
    return scratchpad_size;
}

void BufferSolver::enumerate(const lowered::LinearIR& linear_ir) {
    int64_t order = 0;
    for (const auto& expr : linear_ir) {
        ov::snippets::pass::SetTopologicalOrder(expr->get_node(), order++);
    }
}

void BufferSolver::identify_buffers(lowered::LinearIR& linear_ir) {
    if (ReusingBufferIDBit & m_mode) {
        lowered::pass::IdentifyBuffers pass;
        pass.run(linear_ir);
        return;
    }

    size_t buffer_id = 0;
    for (const auto& expr : linear_ir) {
        const auto op = expr->get_node();
        if (const auto buffer = ov::as_type_ptr<op::Buffer>(op)) {
            buffer->set_id(buffer_id++);
        }
    }
}

BufferSolver::BufferClusters BufferSolver::init_clusters(const lowered::LinearIR& linear_ir) {
    return m_mode & InPlaceOneLevelBit || m_mode & InPlaceMultiLevelBit ? init_inplace_clusters(linear_ir)
                                                                        : init_default_clusters(linear_ir);
}

BufferSolver::BufferClusters BufferSolver::init_default_clusters(const lowered::LinearIR& linear_ir) {
    BufferClusters buffer_clusters;
    for (const auto& expr : linear_ir) {
        if (ov::is_type<op::Buffer>(expr->get_node())) {
            buffer_clusters.push_back({expr});
        }
    }
    return buffer_clusters;
}

BufferSolver::BufferClusters BufferSolver::init_inplace_clusters(const lowered::LinearIR& linear_ir) {
    BufferClusters buffer_clusters;
    auto find_cluster = [&buffer_clusters](const ExpressionPtr& target) {
        for (auto it = buffer_clusters.begin(); it != buffer_clusters.end(); ++it) {
            if (it->count(target) > 0)
                return it;
        }
        return buffer_clusters.end();
    };
    auto create_cluster = [&buffer_clusters, &find_cluster](const ExpressionPtr& target_expr, const ExpressionPtr& node_expr) {
        const auto buffer = ov::as_type_ptr<op::Buffer>(target_expr->get_node());
        // Buffer must be explicitly source for the target LoopEnd expr or MemoryAccess op (there aren't other loop between them)
        if (buffer && target_expr->get_loop_ids() == node_expr->get_loop_ids()) {
            const auto cluster_it = find_cluster(target_expr);
            // If Buffer is missed in clusters, create new cluster with the single Buffer node inside
            if (cluster_it == buffer_clusters.cend()) {
                buffer_clusters.push_back(BufferCluster{target_expr});
            }
            return true;
        }
        return false;
    };
    auto get_cluster_buffer_id = [](const BufferCluster& cluster) {
        OPENVINO_ASSERT(!cluster.empty(), "Buffer cluster is empty!");
        const auto id = (ov::as_type_ptr<op::Buffer>(cluster.cbegin()->get()->get_node()))->get_id();
        if (std::all_of(cluster.cbegin(), cluster.cend(),
                        [&id](const ExpressionPtr& expr) { return (ov::as_type_ptr<op::Buffer>(expr->get_node()))->get_id() == id; })) {
            return id;
        }
        return SIZE_MAX;
    };

    for (auto expr_it = linear_ir.cbegin(); expr_it != linear_ir.cend(); ++expr_it) {
        const auto expr = *expr_it;
        const auto op = expr->get_node();
        if (const auto loop_end = ov::as_type_ptr<op::LoopEnd>(op)) {
            const auto ptr_increments = loop_end->get_ptr_increments();
            const auto final_offsets = loop_end->get_finalization_offsets();
            const auto in_count = loop_end->get_input_num();
            const auto out_count = loop_end->get_output_num();
            const auto connectors = expr->get_input_port_connectors();

            std::set<ExpressionPtr> visited_buffers;
            std::unordered_map<ExpressionPtr, std::set<size_t>> input_buffers;
            std::unordered_map<ExpressionPtr, size_t> output_buffers;
            for (size_t i = 0; i < in_count; ++i) {
                const auto source_expr = connectors[i]->get_source().get_expr();
                const auto is_buffer = create_cluster(source_expr, expr);
                if (is_buffer) {
                    // Save as input Buffer
                    const auto ret = input_buffers.insert(std::make_pair(source_expr, std::set<size_t>{ i })).second;
                    if (!ret)
                        input_buffers[source_expr].insert(i);
                }
            }
            for (size_t i = in_count; i < in_count + out_count; ++i) {
                for (const auto& consumer : connectors[i]->get_consumers()) {
                    auto consumer_expr = consumer.get_expr();
                    const auto buffer = ov::as_type_ptr<op::Buffer>(consumer_expr->get_node());
                    // Buffer must be explicitly source for the target LoopEnd expr (there aren't other loop between them)
                    if (buffer && consumer_expr->get_loop_ids() == expr->get_loop_ids()) {
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
                                    const auto cluster_it = find_cluster(input_buffer_expr);
                                    OPENVINO_ASSERT(cluster_it != buffer_clusters.end(), "Buffer on inputs of Loop must be already saved in clusters");
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
                            buffer_clusters.push_back(BufferCluster{consumer_expr});
                        }
                        output_buffers[consumer_expr] = i;
                    }
                }
            }

            if ((m_mode & InPlaceMultiLevelBit) && (!input_buffers.empty() || !output_buffers.empty())) {
                const auto loop_begin = loop_end->get_loop_begin();
                for (auto it = std::reverse_iterator<LinearIR::constExprIt>(expr_it); (*it)->get_node() != loop_begin; ++it) {
                    const auto inner_expr = *it;
                    if (const auto inner_buffer = ov::as_type_ptr<op::Buffer>(inner_expr->get_node())) {
                        auto inner_cluster_it = find_cluster(inner_expr);
                        OPENVINO_ASSERT(inner_cluster_it != buffer_clusters.cend(), "Buffer cluster has not been found");
                        const auto inner_cluster_id = get_cluster_buffer_id(*inner_cluster_it);
                        if (inner_cluster_id == SIZE_MAX) continue;

                        std::set<int64_t> final_offsets;
                        // life finish time - order of LoopEnd / MemoryAccess ops
                        const auto buffer_outs = inner_expr->get_output_port_connectors();
                        for (const auto& buffer_out : buffer_outs) {
                            const auto consumers = buffer_out->get_consumers();
                            for (const auto& consumer : consumers) {
                                const auto consumer_expr = consumer.get_expr();
                                const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(consumer_expr->get_node());
                                if (loop_end && consumer_expr->get_loop_ids() == inner_expr->get_loop_ids()) {
                                    const auto loop_inputs = consumer_expr->get_input_port_connectors();
                                    final_offsets.insert(
                                        loop_end->get_finalization_offsets()[
                                            std::distance(loop_inputs.cbegin(), std::find(loop_inputs.cbegin(), loop_inputs.cend(), buffer_out))]);
                                }
                            }
                        }

                        for (const auto& in : input_buffers) {
                            const auto cluster_it = find_cluster(in.first);
                            OPENVINO_ASSERT(cluster_it != buffer_clusters.cend(), "Buffer cluster has not been found");
                            if (cluster_it == inner_cluster_it) continue;

                            if (std::none_of(cluster_it->cbegin(), cluster_it->cend(),
                                             [inner_cluster_id](const ExpressionPtr& expr)
                                             { return (ov::as_type_ptr<op::Buffer>(expr->get_node()))->get_id() == inner_cluster_id; } )) {
                                continue;
                            }

                            bool can_be_reused = true;
                            for (const auto idx : in.second) {
                                const auto ptr_increment = loop_end->get_ptr_increments()[idx];
                                can_be_reused = can_be_reused && (ptr_increment != 0);
                                can_be_reused = can_be_reused &&
                                    std::all_of(final_offsets.cbegin(), final_offsets.cend(),
                                                [&ptr_increment](int64_t offset) { return (offset * -1) == ptr_increment; });
                            }
                            if (!can_be_reused)
                                continue;

                            cluster_it->insert(inner_cluster_it->cbegin(), inner_cluster_it->cend());
                            buffer_clusters.erase(inner_cluster_it);
                            inner_cluster_it = buffer_clusters.end();
                            break;
                        }
                        if (inner_cluster_it == buffer_clusters.end()) continue;

                        for (const auto& out : output_buffers) {
                            const auto cluster_it = find_cluster(out.first);
                            OPENVINO_ASSERT(cluster_it != buffer_clusters.cend(), "Buffer cluster has not been found");
                            if (cluster_it == inner_cluster_it) continue;

                            if (std::none_of(cluster_it->cbegin(), cluster_it->cend(),
                                             [inner_cluster_id](const ExpressionPtr& expr)
                                             { return (ov::as_type_ptr<op::Buffer>(expr->get_node()))->get_id() == inner_cluster_id; } )) {
                                continue;
                            }
                            const auto ptr_increment = loop_end->get_ptr_increments()[out.second];
                            if (ptr_increment == 0 ||
                                !std::all_of(final_offsets.cbegin(), final_offsets.cend(),
                                            [&ptr_increment](int64_t offset) { return (offset * -1) == ptr_increment; }))
                                continue;

                            cluster_it->insert(inner_cluster_it->cbegin(), inner_cluster_it->cend());
                            buffer_clusters.erase(inner_cluster_it);
                            break;
                        }
                    }
                }
            }

            continue;
        }
        // TODO: Some full MemoryAccess ops can have inplace inputs and outputs in general.
        //       Need to add mechanism of inplace ports using MemoryAccess::PortDescriptor::inplace
        if (const auto ma = ov::as_type_ptr<op::MemoryAccess>(op)) {
            if (ma->is_full_memory_access_op()) {
                const auto target_loop_ids = expr->get_loop_ids();
                for (const auto& input : expr->get_input_port_connectors()) {
                    create_cluster(input->get_source().get_expr(), expr);
                }
                for (const auto& output : expr->get_output_port_connectors()) {
                    for (const auto& consumer : output->get_consumers()) {
                        create_cluster(consumer.get_expr(), expr);
                    }
                }
            }
        }
    }

    return buffer_clusters;
}

int64_t BufferSolver::allocate(const BufferClusters& buffer_clusters) {
    if (m_mode & MemorySolverBit) {
        const auto boxes = init_boxes(buffer_clusters);
        MemorySolver memSolver(boxes);
        const auto scratchpad_size = static_cast<size_t>(memSolver.solve()) * m_alignment;  // alignment in byte

        // Set offsets for Buffers
        for (const auto& box : boxes) {
            for (const auto& buffer : buffer_clusters[box.id]) {
                int64_t offset = memSolver.getOffset(box.id);
                set_buffer_offset(buffer, offset * m_alignment);  // alignment in byte
            }
        }
        return scratchpad_size;
    }

    size_t buffer_offset = 0;
    for (const auto& cluster : buffer_clusters) {
        size_t cluster_size = 0;
        for (const auto& buffer_expr : cluster) {
            const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
            OPENVINO_ASSERT(buffer != nullptr, "BufferSolver expects Buffers in clusters");
            set_buffer_offset(buffer_expr, buffer_offset);
            cluster_size = std::max(cluster_size, buffer->get_byte_size());
        }
        buffer_offset += cluster_size;
    }
    return static_cast<int64_t>(buffer_offset);
}

std::vector<MemorySolver::Box> BufferSolver::init_boxes(const BufferClusters& buffer_clusters) {
    std::vector<MemorySolver::Box> boxes;
    const auto count = static_cast<int>(buffer_clusters.size());
    for (int i = 0; i < count; i++) {
        MemorySolver::Box box = { std::numeric_limits<int>::max(), 0, 0, i };
        int64_t box_size = 0;
        for (const auto& buffer_expr : buffer_clusters[i]) {
            int e_start = 0, e_finish = 0;
            const auto buffer = ov::as_type_ptr<ov::snippets::op::Buffer>(buffer_expr->get_node());
            OPENVINO_ASSERT(buffer != nullptr, "BufferSolver expects Buffer ops in clusters");
            const auto buffer_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(buffer));

            // life finish time - order of LoopEnd / MemoryAccess ops
            const auto buffer_outs = buffer_expr->get_output_port_connectors();
            for (const auto& buffer_out : buffer_outs) {
                const auto consumers = buffer_out->get_consumers();
                for (const auto& consumer : consumers) {
                    const auto consumer_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(consumer.get_expr()->get_node()));
                    e_finish = std::max(e_finish, consumer_order);
                }
            }
            e_start = e_finish;

            const auto buffer_ins = buffer_expr->get_input_port_connectors();
            for (const auto& buffer_in : buffer_ins) {
                const auto& source = buffer_in->get_source();
                auto local_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(source.get_expr()->get_node()));

                const auto buffer_siblings = buffer_in->get_consumers();
                for (const auto& sibling : buffer_siblings) {
                    const auto loop_end = ov::as_type_ptr<ov::snippets::op::LoopEnd>(sibling.get_expr()->get_node());
                    if (!loop_end)
                        continue;
                    const auto loop_end_order = static_cast<int>(ov::snippets::pass::GetTopologicalOrder(loop_end));
                    if (loop_end_order < buffer_order) {
                        local_order = std::min(local_order, static_cast<int>(ov::snippets::pass::GetTopologicalOrder(loop_end->get_loop_begin())));
                    }
                }
                e_start = std::min(e_start, local_order);
            }

            // TODO: Added support of Dynamic Buffers
            auto buffer_size = static_cast<int64_t>(buffer->get_byte_size());
            box_size = std::max(buffer_size, box_size);

            box.start = std::min(e_start, box.start);
            box.finish = std::max(e_finish, box.finish);
        }

        box.size = utils::div_up(box_size, m_alignment);
        boxes.push_back(box);
    }
    return boxes;
}

void BufferSolver::set_buffer_offset(const ExpressionPtr& buffer_expr, const size_t offset) {
    // If Buffer has offset We set this offset in the connected MemoryAccess ops
    // to correctly read and write data because all Buffers have the common data pointer on buffer scratchpad

    const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
    buffer->set_offset(static_cast<int64_t>(offset));

    // Propagate to up: in Store. Buffer can have only one Store
    {
        if (buffer->is_intermediate_memory()) {
            OPENVINO_ASSERT(buffer_expr->get_input_port_connectors().size() == 1, "Buffer with intermediate memory must have one parent");
            const auto& parent_output = buffer_expr->get_input_port_connector(0)->get_source();
            const auto& parent_expr = parent_output.get_expr();
            const auto port = parent_output.get_index();
            const auto& parent_node = parent_expr->get_node();
            auto memory_access = ov::as_type_ptr<ov::snippets::op::MemoryAccess>(parent_node);
            if (memory_access && memory_access->is_memory_access_output_port(port)) {
                memory_access->set_output_offset(offset, port);
            } else {
                OPENVINO_THROW(
                        "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
            }
        }
    }
    // Propagate to down: in Load. Buffer can have several Load
    const auto& buffer_out = buffer_expr->get_output_port_connector(0);
    for (const auto& child_expr_input : buffer_out->get_consumers()) {
        const auto& child_expr = child_expr_input.get_expr();
        const auto port = child_expr_input.get_index();
        const auto& child_node = child_expr->get_node();
        auto memory_access = ov::as_type_ptr<ov::snippets::op::MemoryAccess>(child_node);
        if (memory_access && memory_access->is_memory_access_input_port(port)) {
            memory_access->set_input_offset(offset, port);
        } else if (ov::is_type<op::LoopEnd>(child_node)) {
            // After Loop initialization, Buffer can be connected to LoopEnd - it's ok
            continue;
        } else {
            OPENVINO_THROW(
                    "Buffer::set_offset() was called when Buffer didn't have the corresponding MemoryAccess op for offset propagation");
        }
    }
}

} // namespace lowered
} // namespace snippets
} // namespace ov
