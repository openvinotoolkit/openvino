// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/runtime_configurator.hpp"

#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_specific_iterations.hpp"
#include "snippets/lowered/pass/compute_buffer_allocation_size.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {

namespace {
void init_data_ptr_shifts(const lowered::UnifiedLoopInfoPtr& unified_loop_info, std::vector<int64_t>& ptr_increments,
                          std::vector<int64_t>& finalization_offsets) {
    const auto count = unified_loop_info->get_input_count() + unified_loop_info->get_output_count();
    ptr_increments.resize(count);
    finalization_offsets.resize(count);

    size_t idx = 0;
    unified_loop_info->iterate_through_descs(
        [&ptr_increments, &finalization_offsets, &idx](const lowered::UnifiedLoopInfo::LoopPortDesc& desc) {
            ptr_increments[idx] = desc.ptr_increment;
            finalization_offsets[idx] = desc.finalization_offset;
            ++idx;
        });
}
}  // namespace

RuntimeConfigurator::RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c) : m_config(std::move(c)) {
    OPENVINO_ASSERT(m_config, "Runtime config is nullptr!");
}

const std::shared_ptr<RuntimeConfig>& RuntimeConfigurator::get_updated_config(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    // First initialization
    if (m_io_num == 0)
        initialization(linear_ir);

    update(linear_ir);
    return m_config;
}

void RuntimeConfigurator::initialization(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    init_data_info(linear_ir);
    init_tensor_rank(linear_ir);
    init_buffer_info(linear_ir);

    OPENVINO_ASSERT(m_io_num > 0, "LinearIR must have parameters and results");
    m_latest_shapes.resize(m_io_num);
    m_config->io_data_offsets.resize(m_io_num);
    m_config->tile_rank = linear_ir->get_config().m_loop_depth;
}

void RuntimeConfigurator::update(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    if (linear_ir->is_dynamic()) {
        update_loop_info(linear_ir);
        update_buffer_scratchpad_size(linear_ir);
    }

    m_config->master_shape = linear_ir->get_master_shape();

    update_data_offsets();
    update_latest_shapes();
}

void RuntimeConfigurator::init_tensor_rank(const std::shared_ptr<lowered::LinearIR>& linear_ir) const {
    m_config->tensor_rank = linear_ir->get_master_shape().size();
}

void RuntimeConfigurator::init_data_info(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    const auto& parameters = linear_ir->get_parameters();
    const auto& results = linear_ir->get_results();
    m_in_num = parameters.size();
    m_io_num = m_in_num + results.size();
    m_io_descs.reserve(m_io_num);
    m_io_data_sizes.reserve(m_io_num);

    auto update_io_parameters = [&](const snippets::lowered::PortDescriptorPtr& desc, const ov::element::Type& etype) {
        OPENVINO_ASSERT(desc, "IO Descriptor is missed!");
        OPENVINO_ASSERT(desc->get_shape().size() == desc->get_layout().size() || desc->get_layout().empty(),
                        "Incompatible ranks of shape and layout!");
        m_io_descs.push_back(desc);
        m_io_data_sizes.push_back(etype.size());
    };

    for (const auto& param : parameters) {
        // input->shape changing ops->load
        snippets::lowered::PortDescriptorPtr desc = nullptr;
        const auto& shape_infer_seq = ov::snippets::utils::get_first_child_shape_infer_expr_seq(param);
        const auto& mem_desc_expr = shape_infer_seq.empty() ? param : shape_infer_seq.back();
        auto consumer_inputs = mem_desc_expr->get_output_port_connector(0)->get_consumers();
        for (const auto& child_input : consumer_inputs) {
            const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(child_input.get_expr()->get_node());
            if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                desc = child_input.get_descriptor_ptr();
                break;
            }
        }
        const auto& etype = mem_desc_expr->get_node()->get_output_element_type(0);
        update_io_parameters(desc, etype);
    }
    for (const auto& result : results) {
        // store->shape changing ops->result
        const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(result);
        const auto& mem_desc_expr = shape_infer_seq.empty() ? result : shape_infer_seq.back();
        const auto& desc = mem_desc_expr->get_input_port_connector(0)->get_source().get_descriptor_ptr();
        const auto& etype = mem_desc_expr->get_node()->get_input_element_type(0);
        update_io_parameters(desc, etype);
    }
}

void RuntimeConfigurator::init_buffer_info(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    std::map<size_t, std::set<lowered::ExpressionPtr>> dynamic_buffer_clusters, static_buffer_clusters;

    // All needed checks are in Validate pass
    const auto& buffer_expressions = linear_ir->get_buffers();
    for (const auto& buffer_expr : buffer_expressions) {
        const auto buffer = ov::as_type_ptr<op::Buffer>(buffer_expr->get_node());
        OPENVINO_ASSERT(buffer, "Expected Buffer ops in Buffer expressions of LinearIR");

        auto& clusters = buffer->is_defined() ? static_buffer_clusters : dynamic_buffer_clusters;
        clusters[buffer->get_cluster_id()].insert(buffer_expr);
    }

    const auto cluster_count = dynamic_buffer_clusters.size() + static_buffer_clusters.size();
    m_config->buffer_scratchpad_size = linear_ir->get_static_buffer_scratchpad_size();
    m_config->buffer_cluster_offsets.resize(cluster_count, utils::get_dynamic_value<size_t>());

    for (const auto& p : static_buffer_clusters) {
        const auto& cluster_id = p.first;
        const auto& cluster = p.second;

        OPENVINO_ASSERT(cluster.size() > 0, "Incorrect size of buffer cluster");
        size_t cluster_offset = ov::as_type_ptr<op::Buffer>((*cluster.cbegin())->get_node())->get_offset();
        m_config->buffer_cluster_offsets[cluster_id] = cluster_offset;
    }

    m_dynamic_buffer_clusters = std::move(dynamic_buffer_clusters);
}

void RuntimeConfigurator::update_loop_info(const std::shared_ptr<lowered::LinearIR>& linear_ir) const {
    // Initialized UnifiedLoopInfo
    struct CurrentUnifiedLoopInfo {
        size_t current_work_amount = 0;
        std::vector<int64_t> ptr_increments;
        std::vector<int64_t> finalization_offsets;
    };
    std::unordered_map<lowered::UnifiedLoopInfoPtr, CurrentUnifiedLoopInfo> initializated_info_map;

    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    for (const auto& p : loop_map) {
        const auto& expanded_loop_info = ov::as_type_ptr<lowered::ExpandedLoopInfo>(p.second);
        OPENVINO_ASSERT(expanded_loop_info, "UpdateLoopInfo expects ExpandedLoopInfo in LoopManager");

        // First visiting of unified (whole) loop
        const auto& current_unified_loop_info = expanded_loop_info->get_unified_loop_info();
        if (initializated_info_map.count(current_unified_loop_info) == 0) {
            auto& current_info = initializated_info_map[current_unified_loop_info];
            lowered::pass::InitLoops::init_loop_info(current_unified_loop_info, true);

            current_info.current_work_amount = current_unified_loop_info->get_work_amount();
            init_data_ptr_shifts(current_unified_loop_info, current_info.ptr_increments, current_info.finalization_offsets);
        }

        auto& initializated_info = initializated_info_map.at(current_unified_loop_info);
        auto& current_work_amount = initializated_info.current_work_amount;
        const auto& ptr_increments = initializated_info.ptr_increments;
        const auto& finalization_offsets = initializated_info.finalization_offsets;

        const auto& decomposed_loop_type = expanded_loop_info->get_type();

        // If the specific iteration is not needed, we skip loop evaluation - set zero as work amount is enough
        if (!lowered::pass::InsertSpecificIterations::is_decomposed_loop_needed(current_unified_loop_info, decomposed_loop_type, current_work_amount)) {
            expanded_loop_info->set_work_amount(0);
            continue;
        }

        expanded_loop_info->set_work_amount(
            lowered::pass::InsertSpecificIterations::get_decomposed_loop_work_amount(current_unified_loop_info, decomposed_loop_type, current_work_amount));
        // Update remaining Loop work amount
        current_work_amount -= expanded_loop_info->get_work_amount();

        expanded_loop_info->update_ptr_increments(ptr_increments);
        if (current_work_amount > 0) {
            expanded_loop_info->update_finalization_offsets(std::vector<int64_t>(finalization_offsets.size(), 0));
        } else {
            expanded_loop_info->update_finalization_offsets(finalization_offsets);
        }
    }
}

void RuntimeConfigurator::update_buffer_scratchpad_size(const std::shared_ptr<lowered::LinearIR>& linear_ir) const {
    const auto& loop_manager = linear_ir->get_loop_manager();
    m_config->buffer_scratchpad_size = linear_ir->get_static_buffer_scratchpad_size();

    for (const auto& p : m_dynamic_buffer_clusters) {
        const auto& cluster_id = p.first;
        const auto& cluster = p.second;

        auto& cluster_offset = m_config->buffer_cluster_offsets[cluster_id];
        cluster_offset = utils::get_dynamic_value<size_t>();

        size_t additional_size = 0;
        for (const auto& buffer_expr : cluster) {
            const auto& allocation_size = lowered::pass::ComputeBufferAllocationSize::get_allocation_size(loop_manager, buffer_expr, m_config->tile_rank);
            additional_size = std::max(allocation_size * buffer_expr->get_node()->get_element_type().size(), additional_size);
        }

        cluster_offset = m_config->buffer_scratchpad_size;
        OPENVINO_ASSERT(!utils::is_dynamic_value(cluster_offset), "Offset of the cluster must be defined!");
        OPENVINO_ASSERT(!utils::is_dynamic_value(additional_size), "Buffer scratchpad size must be defined!");
        m_config->buffer_scratchpad_size += additional_size;
    }

    OPENVINO_ASSERT(!utils::is_dynamic_value(m_config->buffer_scratchpad_size), "Buffer scratchpad size must be defined!");
}

void RuntimeConfigurator::update_data_offsets() const {
    for (size_t i = 0; i < m_io_num; ++i) {
        // offsets represent distance between consecutive elements of corresponding dimension.
        // If a dim size == 1, then the next dim starts immediately and the stride is 0
        // case 1:
        //    shape:         s0,    s1, s2, s3
        //    offsets: s1*s2*s3, s2*s3, s3,  1
        // case 2:
        //    shape:      s0, s1, s2 == 1, s3
        //    offsets: s1*s3, s3,       0,  1
        const auto& shape = m_io_descs[i]->get_shape();
        if (shape == m_latest_shapes[i])
            continue;

        const auto& layout = m_io_descs[i]->get_layout();
        auto& offsets = m_config->io_data_offsets[i];

        offsets.resize(m_config->tensor_rank);
        std::fill(offsets.begin(), offsets.end(), 0);
        if (ov::snippets::utils::is_dynamic_vdims(shape))
            return;

        size_t dim_step = m_io_data_sizes[i];
        offsets[offsets.size() - 1] = dim_step;

        OPENVINO_ASSERT(m_config->tensor_rank >= shape.size(), "Incorrect tensor rank!");
        const auto idx_stride = m_config->tensor_rank - shape.size();
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
            dim_step *= shape[i + 1];
            offsets[i + idx_stride] = shape[i] != 1 ? dim_step : 0;
        }
        if (!layout.empty()) {
            std::vector<size_t> reordered_offsets(offsets.size());
            const auto is_input = i < m_in_num;
            for (size_t i = 0; i < layout.size(); i++) {
                const auto& src_idx = is_input ? layout[i] : i;
                const auto& dst_idx = is_input ? i : layout[i];
                reordered_offsets[idx_stride + dst_idx] = offsets[idx_stride + src_idx];
            }
            offsets = std::move(reordered_offsets);
        }
    }
}

void RuntimeConfigurator::update_latest_shapes() {
    for (size_t i = 0; i < m_io_num; ++i) {
        m_latest_shapes[i] = m_io_descs[i]->get_shape();
    }
}

} // namespace snippets
} // namespace ov
