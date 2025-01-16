// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov {
namespace intel_cpu {

BrgemmExternalRepackingAdjuster::BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                                 const CPURuntimeConfigurator* configurator)
    : snippets::lowered::pass::RuntimeOptimizer(configurator) {
    const auto& params = linear_ir->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto consumers = param->get_output_port_connector(0)->get_consumers();
        const bool brgemm_with_extracted_repacking =
            std::any_of(consumers.begin(), consumers.end(), [](const ov::snippets::lowered::ExpressionPort& port) {
                auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(port.get_expr()->get_node());
                return brgemm && brgemm_utils::with_repacking(brgemm->get_type()) && port.get_index() == 1;
            });
        if (brgemm_with_extracted_repacking) {
            m_param_idces_with_external_repacking.insert(i);
            // Ticket 157339: Support non-planar layout
            OPENVINO_ASSERT(ov::snippets::utils::is_planar_layout(configurator->get_io_descs()[i]->get_layout()),
                            "Non-planar layout is not supported for external repacking");
        }
    }
}

bool BrgemmExternalRepackingAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmExternalRepackingAdjuster")
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    auto& optimal_descs = cpu_config->m_in_requested_descs;
    for (const auto& i : m_param_idces_with_external_repacking) {
        const auto& shape = cpu_config->io_shapes[i];
        const auto& K = *++shape.rbegin();
        const auto& N = *shape.rbegin();

        const auto& precision = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        const auto vnni_factor = brgemm_utils::compute_vnni_factor(precision);
        const size_t brgemm_kernel_rank = 2;
        // Firstly, batch dims are set
        VectorDims requested_blocked_shape(shape.begin(), shape.end() - brgemm_kernel_rank);
        // Then, the blocked dims are formed
        requested_blocked_shape.insert(
            requested_blocked_shape.end(),
            {snippets::utils::div_up(K, vnni_factor), std::max(N, brgemm_utils::repacking::compute_inner_n_block(precision)), vnni_factor});

        VectorDims requested_order(shape.size() - brgemm_kernel_rank);
        std::iota(requested_order.begin(), requested_order.end(), 0);
        const auto last_idx = shape.size() - 1;
        requested_order.insert(requested_order.end(), {last_idx - 1, last_idx, last_idx - 1});

        optimal_descs[i] = std::make_shared<CpuBlockedMemoryDesc>(precision, Shape(shape), requested_blocked_shape, requested_order);

        ov::snippets::VectorDims shape_for_offset(cpu_config->tensor_rank - shape.size(), 1);
        shape_for_offset.insert(shape_for_offset.end(), requested_blocked_shape.begin(), requested_blocked_shape.end());
        m_configurator->compute_offsets(shape_for_offset, i, 0);
    }
    return true;
}

}   // namespace intel_cpu
}   // namespace ov
