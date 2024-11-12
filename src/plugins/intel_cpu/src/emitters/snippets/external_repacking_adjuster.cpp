// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "snippets/utils/utils.hpp"

#ifndef OPENVINO_ARCH_ARM64
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#endif

namespace ov {
namespace intel_cpu {

#ifdef OPENVINO_ARCH_ARM64
BrgemmExternalRepackingAdjuster::BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
}

void BrgemmExternalRepackingAdjuster::optimize() {
}
#else
BrgemmExternalRepackingAdjuster::BrgemmExternalRepackingAdjuster(
    const ov::snippets::lowered::LinearIRCPtr& linear_ir,
    CPURuntimeConfigurator* configurator) : m_configurator(configurator) {
    const auto& params = linear_ir->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto consumers = param->get_output_port_connector(0)->get_consumers();
        const bool brgemm_with_extracted_repacking =
            std::any_of(consumers.begin(), consumers.end(), [](const ov::snippets::lowered::ExpressionPort& port) {
                auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(port.get_expr()->get_node());
                return port.get_index() == 1 && brgemm && brgemm_utils::with_repacking(brgemm->get_type());
            });
        if (brgemm_with_extracted_repacking) {
            m_param_idces_with_external_repacking.insert(i);
        }
    }
}

void BrgemmExternalRepackingAdjuster::optimize(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    auto& optimal_descs = cpu_config->m_in_requested_descs;
    for (const auto& i : m_param_idces_with_external_repacking) {
        const auto& shape = m_configurator->get_config()->shapes[i];
        // TODO: support orbitrary order
        const auto& K = *++shape.rbegin();
        const auto& N = *shape.rbegin();

        const auto& precision = linear_ir->get_parameters()[i]->get_node()->get_output_element_type(0);
        const auto vnni_factor = brgemm_utils::compute_vnni_factor(precision);
        // Firstly, batch dims are set
        VectorDims requested_blocked_shape(shape.begin(), shape.end() - cpu_config->tile_rank);
        // Then, the blocked dims are formed
        requested_blocked_shape.insert(
            requested_blocked_shape.end(),
            {snippets::utils::div_up(K, vnni_factor), std::max(N, brgemm_utils::repacking::compute_inner_n_block(precision)), vnni_factor});

        VectorDims requested_order(shape.size() - cpu_config->tile_rank);
        std::iota(requested_order.begin(), requested_order.end(), 0);
        const auto last_idx = shape.size() - 1;
        requested_order.insert(requested_order.end(), {last_idx - 1, last_idx, last_idx - 1});

        optimal_descs[i] = std::make_shared<CpuBlockedMemoryDesc>(precision, Shape(shape), requested_blocked_shape, requested_order);

        ov::snippets::VectorDims shape_for_offset(cpu_config->tensor_rank - shape.size(), 1);
        shape_for_offset.insert(shape_for_offset.end(), requested_blocked_shape.begin(), requested_blocked_shape.end());
        auto& offsets = cpu_config->io_data_offsets[i];
        snippets::RuntimeConfigurator::compute_offsets(shape_for_offset, offsets, shape_for_offset.size(), m_configurator->get_io_data_sizes()[i], 0);
        // TODO: Support non-planar layout
        OPENVINO_ASSERT(ov::snippets::utils::is_planar_layout(m_configurator->get_config()->layouts[i]));
    }
}
#endif

}   // namespace intel_cpu
}   // namespace ov
