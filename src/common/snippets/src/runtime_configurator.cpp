// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/runtime_configurator.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/pass/update_loop_info.hpp"

namespace ov {
namespace snippets {

RuntimeConfigurator::RuntimeConfigurator(std::shared_ptr<RuntimeConfig> c) : m_config(std::move(c)) {
    OPENVINO_ASSERT(m_config, "Runtime config is nullptr!");

    // Init LinearIR StateUpdater: some passes to update LoopInfo, BufferInfo etc
    m_state_updater.register_pass<lowered::pass::UpdateLoopInfo>();
}

const std::shared_ptr<RuntimeConfig>& RuntimeConfigurator::get_updated_config(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    // First initialization
    if (m_io_num == 0)
        initialization(linear_ir);

    update(linear_ir);
    return m_config;
}

void RuntimeConfigurator::update(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    if (linear_ir->is_dynamic()) {
        m_state_updater.run(*linear_ir);
    }

    m_config->master_shape = linear_ir->get_master_shape();
    m_config->buffer_scratchpad_size = linear_ir->get_buffer_scratchpad_size();

    update_data_offsets();
    update_latest_shapes();
}

void RuntimeConfigurator::initialization(const std::shared_ptr<lowered::LinearIR>& linear_ir) {
    init_data_info(linear_ir);
    init_tensor_rank(linear_ir);

    OPENVINO_ASSERT(m_io_num > 0, "LinearIR must have parameters and results");
    m_latest_shapes.resize(m_io_num);
    m_config->io_data_offsets.resize(m_io_num);
    m_config->tile_rank = linear_ir->get_config().m_loop_depth;
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
