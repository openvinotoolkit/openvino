// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"


namespace ov {
namespace intel_cpu {

CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {}

bool CPURuntimeConfigurator::is_update_needed(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    if (m_latest_input_shapes.empty())
        return true;
    for (size_t i = 0; i < m_latest_input_shapes.size(); ++i)
        if (m_latest_input_shapes[i] != m_io_descs[i]->get_shape())
            return true;
    return false;
}

void CPURuntimeConfigurator::update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_config);
    OPENVINO_ASSERT(cpu_config, "CPURuntimeConfigurator expects CPURuntimeConfig");

    if (m_io_num == 0) {
        init_data_info(linear_ir);
        cpu_config->tensor_rank = std::max(linear_ir->get_master_shape().size(), rank6D);
    }

    if (linear_ir->is_dynamic()) {
        update_linear_ir_state(linear_ir);
        update_loop_args(linear_ir, cpu_config);
    }

    update_data_offsets(cpu_config);
    update_parallel_domain(linear_ir, cpu_config);

    update_latest_shapes();
}

void CPURuntimeConfigurator::init_data_info(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    const auto& io_exprs = linear_ir->get_IO_ops();
    m_io_num = io_exprs.size();
    m_io_descs.resize(m_io_num);
    m_io_data_sizes.resize(m_io_num);
    m_latest_input_shapes.resize(m_io_num);
    m_in_num = 0;

    size_t idx = 0;
    for (const auto& expr : io_exprs) {
        switch (expr->get_type()) {
            case ov::snippets::lowered::IOExpression::io_type::INPUT: {
                // input->shape changing ops->load
                const auto& shape_infer_seq = ov::snippets::utils::get_first_child_shape_infer_expr_seq(expr);
                const auto& mem_desc_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
                auto consumer_inputs = mem_desc_expr->get_output_port_connector(0)->get_consumers();
                for (const auto& child_input : consumer_inputs) {
                    const auto ma = std::dynamic_pointer_cast<snippets::modifier::MemoryAccess>(child_input.get_expr()->get_node());
                    if (ma && ma->is_memory_access_input_port(child_input.get_index())) {
                        m_io_descs[idx] = child_input.get_descriptor_ptr();
                        break;
                    }
                }
                m_io_data_sizes[idx] = mem_desc_expr->get_node()->get_output_element_type(0).size();
                m_in_num++;
                break;
            }
            case ov::snippets::lowered::IOExpression::io_type::OUTPUT: {
                const auto& shape_infer_seq = ov::snippets::utils::get_first_parent_shape_infer_expr_seq(expr);
                const auto& mem_desc_expr = shape_infer_seq.empty() ? expr : shape_infer_seq.back();
                const auto& parent_output = mem_desc_expr->get_input_port_connector(0)->get_source();
                m_io_descs[idx] = parent_output.get_descriptor_ptr();
                m_io_data_sizes[idx] = expr->get_node()->get_input_element_type(0).size();
                break;
            } default : {
                OPENVINO_THROW("Detected unsupported io_type");
            }
        }
        OPENVINO_ASSERT(m_io_descs[idx], "IO Descriptor is missed!");
        OPENVINO_ASSERT(m_io_descs[idx]->get_shape().size() == m_io_descs[idx]->get_layout().size() || m_io_descs[idx]->get_layout().size() == 0,
                        "Incompatible ranks of shape and layout!");
        idx++;
    }
}


void CPURuntimeConfigurator::update_data_offsets(const std::shared_ptr<CPURuntimeConfig>& cpu_config) const {
    cpu_config->io_data_offsets.resize(m_io_num);
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
        if (shape == m_latest_input_shapes[i])
            continue;

        const auto& layout = m_io_descs[i]->get_layout();
        auto& offsets = cpu_config->io_data_offsets[i];

        offsets.resize(cpu_config->tensor_rank);
        std::fill(offsets.begin(), offsets.end(), 0);
        if (ov::snippets::utils::is_dynamic_vdims(shape))
            return;

        size_t dim_step = m_io_data_sizes[i];
        offsets[offsets.size() - 1] = dim_step;

        OPENVINO_ASSERT(cpu_config->tensor_rank >= shape.size(), "Incorrect tensor rank!");
        const auto idx_stride = cpu_config->tensor_rank - shape.size();
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

void CPURuntimeConfigurator::update_loop_args(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                              const std::shared_ptr<CPURuntimeConfig>& cpu_config) const {
    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    cpu_config->loop_args.resize(loop_map.size());
    for (const auto& loop : loop_map) {
        const auto& idx = loop.first;
        const auto& loop_info = std::dynamic_pointer_cast<ov::snippets::lowered::ExpandedLoopInfo>(loop.second);
        OPENVINO_ASSERT(loop_info, "CPURuntimeConfigurator expects ExpandedLoopInfo in loop manager");

        const auto& increment = loop_info->get_increment();
        const auto& data_sizes = loop_info->get_data_sizes();

        auto& loop_arg = cpu_config->loop_args[idx];
        loop_arg = jit_snippets_call_args::loop_args_t(loop_info->get_work_amount(), loop_info->get_ptr_increments(), loop_info->get_finalization_offsets());
        for (int64_t i = 0; i < loop_arg.m_num_data_ptrs; ++i) {
            loop_arg.m_ptr_increments[i] *= (increment * data_sizes[i]);
            loop_arg.m_finalization_offsets[i] *= data_sizes[i];
        }
    }
}

void CPURuntimeConfigurator::update_parallel_domain(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir,
                                                    const std::shared_ptr<CPURuntimeConfig>& cpu_config) const {
    cpu_config->parallel_domain.resize(cpu_config->tensor_rank, 1);
    const auto parallel_exec_domain = linear_ir->get_parallel_domain();
    std::copy(parallel_exec_domain.cbegin(), parallel_exec_domain.cend(),
              cpu_config->parallel_domain.begin() + (cpu_config->tensor_rank - parallel_exec_domain.size()));
}

void CPURuntimeConfigurator::update_latest_shapes() {
    m_latest_input_shapes.resize(m_in_num);
    for (size_t i = 0; i < m_in_num; ++i) {
        m_latest_input_shapes[i] = m_io_descs[i]->get_shape();
    }
}

} // namespace intel_cpu
} // namespace ov
