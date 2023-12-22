// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/runtime_configurator.hpp"

#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"


namespace ov {
namespace snippets {
namespace lowered {

void RuntimeConfigurator::init_linear_info(const lowered::LinearIR& linear_ir) {
    init_io_info(linear_ir);
}

void RuntimeConfigurator::update(const lowered::LinearIR& linear_ir) {
    if (m_is_first_init) {
        reset();
        init_linear_info(linear_ir);
    }

    const auto& loop_manager = linear_ir.get_loop_manager();
    init_loop_descriptors(loop_manager);
    init_data_offsets(linear_ir);

    m_is_first_init = false;
}

void RuntimeConfigurator::reset() {
    m_is_first_init = true;
    m_config.clear();
}

void RuntimeConfigurator::init_loop_descriptors(const lowered::LinearIR::LoopManagerPtr& loop_manager) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::init_loop_descriptors")

    const auto& loop_map = loop_manager->get_map();
    for (const auto& loop_pair : loop_map) {
        const auto loop_id = loop_pair.first;
        // make a copy to avoid original loop info corruption
        const auto loop_info = std::make_shared<LinearIR::LoopManager::LoopInfo>(*loop_pair.second);
        lowered::pass::InitLoops::init_loop_info(loop_info, true);

        OPENVINO_ASSERT(!utils::is_dynamic_vdim(loop_info->get_increment()), "Increment must be static value!");
        OPENVINO_ASSERT(utils::implication(m_is_first_init, m_config.m_loops.count(loop_id) == 0),
                        "If it's an first initialization, there should not be loop descriptors");
        if (m_is_first_init) m_config.m_loops[loop_id] = {};

        if (is_first_iter_loop_needed(loop_info)) {
            auto desc_it = m_is_first_init ? m_config.push_new_desc(loop_id, RuntimeConfig::LoopDescriptor::First)
                                           : m_config.get_loop_desc_it(loop_id, RuntimeConfig::LoopDescriptor::First);
            OPENVINO_ASSERT(desc_it != m_config.m_loops.at(loop_id).end(), "First Loop Descriptor has not been found!");
            init_first_iter_loop_descriptor(loop_info, loop_id, *desc_it);
        }
        if (is_vector_loop_needed(loop_info)) {
            auto desc_it = m_is_first_init ? m_config.push_new_desc(loop_id, RuntimeConfig::LoopDescriptor::Vector)
                                           : m_config.get_loop_desc_it(loop_id, RuntimeConfig::LoopDescriptor::Vector);
            OPENVINO_ASSERT(desc_it != m_config.m_loops.at(loop_id).end(), "Vector Loop Descriptor has not been found!");
            init_vector_loop_descriptor(loop_info, loop_id, *desc_it);
        }
        if (is_tail_loop_needed(loop_info)) {
            auto desc_it = m_is_first_init ? m_config.push_new_desc(loop_id, RuntimeConfig::LoopDescriptor::Tail)
                                           : m_config.get_loop_desc_it(loop_id, RuntimeConfig::LoopDescriptor::Tail);
            OPENVINO_ASSERT(desc_it != m_config.m_loops.at(loop_id).end(), "Tail Loop Descriptor has not been found!");
            init_tail_loop_descriptor(loop_info, loop_id, *desc_it);
            // Inner splited Loop update
            init_inner_splited_tail_loop_descriptors(loop_manager, loop_info, *desc_it, loop_id);
        }
    }
}

void RuntimeConfigurator::init_first_iter_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                          RuntimeConfig::LoopDescriptor& first_iter_loop_desc) {
    first_iter_loop_desc.work_amount = utils::is_dynamic_vdim(loop_info->get_work_amount()) ? loop_info->get_work_amount()
                                                                                            : loop_info->get_increment();
    first_iter_loop_desc.increment = loop_info->get_increment();

    init_data_ptr_shifts(first_iter_loop_desc, loop_info, false, false, m_config.m_loops.at(loop_id).end());
}

void RuntimeConfigurator::init_vector_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                      RuntimeConfig::LoopDescriptor& vector_loop_desc) {
    auto first_iter_desc_it = m_config.get_loop_desc_it(loop_id, RuntimeConfig::LoopDescriptor::First);
    const auto first_needed = first_iter_desc_it != m_config.m_loops.at(loop_id).cend() && first_iter_desc_it->work_amount > 0;
    const auto is_wa_dynamic = utils::is_dynamic_vdim(loop_info->get_work_amount());
    const auto target_work_amount = is_wa_dynamic ? loop_info->get_work_amount() :
                                    first_needed ? loop_info->get_work_amount() - first_iter_desc_it->work_amount : loop_info->get_work_amount();
    const auto skip_evaluation = !is_wa_dynamic && target_work_amount < loop_info->get_increment();

    vector_loop_desc.work_amount = skip_evaluation ? 0 : target_work_amount;
    vector_loop_desc.increment = loop_info->get_increment();

    init_data_ptr_shifts(vector_loop_desc, loop_info, skip_evaluation, first_needed, first_iter_desc_it);
}

void RuntimeConfigurator::init_tail_loop_descriptor(const LinearIR::LoopManager::LoopInfoPtr& loop_info, size_t loop_id,
                                                    RuntimeConfig::LoopDescriptor& tail_loop_desc) {
    auto& loop_descs = m_config.m_loops.at(loop_id);
    auto last_execution_loop = loop_descs.end();
    for (auto desc_it = loop_descs.rbegin(); desc_it != loop_descs.rend(); ++desc_it) {
        if (desc_it->type != RuntimeConfig::LoopDescriptor::Tail && !utils::is_dynamic_vdim(desc_it->work_amount) && desc_it->work_amount > 0) {
            last_execution_loop = std::prev(desc_it.base());
            break;
        }
    }
    const auto there_is_before_loop = last_execution_loop != loop_descs.end();
    const auto is_wa_dynamic = utils::is_dynamic_vdim(loop_info->get_work_amount());
    const auto target_work_amount = is_wa_dynamic ? loop_info->get_work_amount() : loop_info->get_work_amount() % loop_info->get_increment();
    const auto skip_evaluation = !is_wa_dynamic && target_work_amount == 0;

    tail_loop_desc.work_amount = target_work_amount;
    tail_loop_desc.increment = loop_info->is_dynamic() ? 1 : tail_loop_desc.work_amount;

    init_data_ptr_shifts(tail_loop_desc, loop_info, skip_evaluation, there_is_before_loop, last_execution_loop);
}

void RuntimeConfigurator::init_inner_splited_tail_loop_descriptors(const LinearIR::LoopManagerPtr& loop_manager,
                                                                   const LinearIR::LoopManager::LoopInfoPtr& outer_splited_loop_info,
                                                                   const RuntimeConfig::LoopDescriptor& outer_splited_tail_loop_desc,
                                                                   size_t outer_loop_id) {
    if (!outer_splited_loop_info->get_outer_splited_loop())
        return;

    const auto is_outer_vector_loop_needed =
        m_config.get_loop_desc_it(outer_loop_id, RuntimeConfig::LoopDescriptor::Vector) != m_config.m_loops.at(outer_loop_id).end();
    const auto tail_size = outer_splited_tail_loop_desc.increment;
    const auto outer_dim_idx = outer_splited_loop_info->get_dim_idx();
    const auto& loop_map = loop_manager->get_map();
    // go through all loops in loop manager to find inner loops by port loop IDs
    for (const auto& p : loop_map) {
        const auto inner_loop_id = p.first;
        const auto inner_loop_info = p.second;
        // skip the current loop
        if (inner_loop_id == outer_loop_id)
            continue;

        // check if the target outer splited loop is really outer loop of the analyzed loop using loop IDs of ports
        OPENVINO_ASSERT(!inner_loop_info->get_entry_points().empty(), "Each Loop must have one entry port at least!");
        const auto loop_port = inner_loop_info->get_entry_points().front();
        const auto outer_loop_ids = LinearIR::LoopManager::get_outer_expr_loops(loop_port.expr_port->get_expr(), inner_loop_id);
        if (std::find(outer_loop_ids.cbegin(), outer_loop_ids.cend(), outer_loop_id) == outer_loop_ids.cend())
            continue;

        // check if the target outer splited loop and the analyzed inner loop have the same dim_index
        const auto inner_dim_idx = inner_loop_info->get_dim_idx();
        if (inner_dim_idx != outer_dim_idx)
            continue;

        auto splited_tail_desc_it = m_is_first_init ? m_config.push_new_desc(inner_loop_id, RuntimeConfig::LoopDescriptor::SplitedTail)
                                                    : m_config.get_loop_desc_it(inner_loop_id, RuntimeConfig::LoopDescriptor::SplitedTail);
        auto inner_vector_dest_it = m_config.get_loop_desc_it(inner_loop_id, RuntimeConfig::LoopDescriptor::Vector);
        OPENVINO_ASSERT(inner_vector_dest_it != m_config.m_loops.at(inner_loop_id).end(), "Splited inner Loop should be already inited!");
        splited_tail_desc_it->work_amount = tail_size;
        splited_tail_desc_it->increment = std::min(inner_vector_dest_it->increment, tail_size);
        splited_tail_desc_it->ptr_increments = inner_vector_dest_it->ptr_increments;
        splited_tail_desc_it->finalization_offsets = inner_vector_dest_it->finalization_offsets;
        // rescale offsets
        for (auto& offset : splited_tail_desc_it->finalization_offsets) {
            offset = offset / static_cast<int64_t>(inner_vector_dest_it->work_amount) * static_cast<int64_t>(splited_tail_desc_it->work_amount);
        }
        // If outer splited loop doesn't have Vector Loop, inner splited loop shouldn't have the Vector Loop as well
        if (!is_outer_vector_loop_needed) {
            m_config.m_loops.at(inner_loop_id).erase(inner_vector_dest_it);
        }
    }
}

void RuntimeConfigurator::init_data_ptr_shifts(RuntimeConfig::LoopDescriptor& desc,
                                               const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                               bool skip_evaluation, bool is_there_prev_iter,
                                               const RuntimeConfig::LoopDescriptorList::iterator& prev_iter_desc) {
    const auto& in_ports = loop_info->get_entry_points();
    const auto& out_ports = loop_info->get_exit_points();
    const auto in_num = in_ports.size();
    const auto out_num = out_ports.size();
    desc.ptr_increments.resize(in_num + out_num);
    desc.finalization_offsets.resize(in_num + out_num);

    if (skip_evaluation) {
        std::fill(desc.ptr_increments.begin(), desc.ptr_increments.end(), 0);
        std::fill(desc.finalization_offsets.begin(), desc.finalization_offsets.end(), 0);
        return;
    }

    const auto& increment = desc.increment;

    auto init_shifts = [&](const std::vector<LinearIR::LoopManager::LoopPort>& loop_ports, size_t start_index) {
        for (size_t i = 0; i < loop_ports.size(); ++i) {
            const auto& loop_port = loop_ports[i];
            desc.ptr_increments[start_index + i] =
                LinearIR::LoopManager::LoopPort::is_dynamic_value(loop_port.ptr_increment) ? loop_port.ptr_increment
                                                                                           : increment * loop_port.ptr_increment * loop_port.data_size;
            if (!is_there_prev_iter)
                desc.finalization_offsets[start_index + i] =
                    LinearIR::LoopManager::LoopPort::is_dynamic_value(loop_port.finalization_offset) ? loop_port.finalization_offset
                                                                                                     : loop_port.finalization_offset * loop_port.data_size;
        }
    };
    init_shifts(in_ports, 0);
    init_shifts(out_ports, in_num);

    if (is_there_prev_iter) {
        desc.finalization_offsets = prev_iter_desc->finalization_offsets;
        std::fill(prev_iter_desc->finalization_offsets.begin(), prev_iter_desc->finalization_offsets.end(), 0);
    }
}

void RuntimeConfigurator::init_io_info(const LinearIR& linear_ir) {
    const auto& io_exprs = linear_ir.get_IO_ops();
    m_io_num = io_exprs.size();
    m_config.m_data_offsets.resize(m_io_num);
    m_io_descs.resize(m_io_num);
    m_io_data_sizes.resize(m_io_num);
    m_in_num = 0;

    size_t idx = 0;
    for (const auto& expr : io_exprs) {
        switch (expr->get_type()) {
            case lowered::IOExpression::io_type::INPUT: {
                // Note that here we consider only the first child (which is usually load),
                // but often there is another child - LoopEnd
                auto consumer_inputs = expr->get_output_port_connector(0)->get_consumers();
                const auto& first_consumer = consumer_inputs.begin()->get_expr();
                // If there is a RankNormalization op after a parameter - we should skip it
                if (is_type<snippets::op::RankNormalization>(first_consumer->get_node()))
                    consumer_inputs = first_consumer->get_output_port_connector(0)->get_consumers();
                // TODO: Add validation pass after control flow pipeline that all consumers have the same layout
                for (const auto& child_input : consumer_inputs) {
                    const auto ma = ov::as_type_ptr<snippets::op::MemoryAccess>(child_input.get_expr()->get_node());
                    if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                        m_io_descs[idx] = child_input.get_descriptor_ptr();
                    }
                }
                m_io_data_sizes[idx] = expr->get_node()->get_output_element_type(0).size();
                m_in_num++;
                break;
            }
            case lowered::IOExpression::io_type::OUTPUT: {
                m_io_descs[idx] = expr->get_input_port_connector(0)->get_source().get_descriptor_ptr();
                m_io_data_sizes[idx] = expr->get_node()->get_input_element_type(0).size();
                break;
            } default : {
                OPENVINO_THROW("Detected unsupported io_type");
            }
        }
        idx++;
    }
}

void RuntimeConfigurator::init_data_offsets(const LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::RuntimeConfig::init_data_offsets")

    const auto& tensor_rank = linear_ir.get_config().m_tensor_rank;

    for (size_t i = 0; i < m_io_num; ++i) {
        offset_calculation(m_io_descs[i], m_io_data_sizes[i], i < m_in_num, tensor_rank, m_config.m_data_offsets[i]);
    }
}

void RuntimeConfigurator::offset_calculation(const lowered::PortDescriptorPtr& desc, size_t data_size, bool is_input,
                                             size_t rank, std::vector<size_t>& offsets) {
    // offsets represent distance between consecutive elements of corresponding dimension.
    // If a dim size == 1, then the next dim starts immediately and the stride is 0
    // case 1:
    //    shape:         s0,    s1, s2, s3
    //    offsets: s1*s2*s3, s2*s3, s3,  1
    // case 2:
    //    shape:      s0, s1, s2 == 1, s3
    //    offsets: s1*s3, s3,       0,  1
    const auto& shape = desc->get_shape();
    const auto& layout = desc->get_layout();

    offsets.resize(rank);
    std::fill(offsets.begin(), offsets.end(), 0);
    if (utils::is_dynamic_vdims(shape))
        return;

    size_t dim_step = data_size;
    offsets[offsets.size() - 1] = dim_step;

    OPENVINO_ASSERT(rank >= shape.size(), "Incorrect tensor rank!");
    const auto idx_stride = rank - shape.size();
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
        dim_step *= shape[i + 1];
        offsets[i + idx_stride] = shape[i] != 1 ? dim_step : 0;
    }
    if (!layout.empty()) {
        std::vector<size_t> reordered_offsets(offsets.size());
        for (size_t i = 0; i < layout.size(); i++) {
            const auto& src_idx = is_input ? layout[i] : i;
            const auto& dst_idx = is_input ? i : layout[i];
            reordered_offsets[idx_stride + dst_idx] = offsets[idx_stride + src_idx];
        }
        offsets = std::move(reordered_offsets);
    }
}

} // namespace lowered
} // namespace snippets
} // namespace ov
