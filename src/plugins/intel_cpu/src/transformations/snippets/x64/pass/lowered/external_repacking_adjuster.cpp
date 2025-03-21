// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu {

const size_t BrgemmExternalRepackingAdjuster::brgemm_kernel_rank = 2;

BrgemmExternalRepackingAdjuster::BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                                 const CPURuntimeConfigurator* configurator)
    : snippets::lowered::pass::RuntimeOptimizer(configurator) {
    const auto& params = linear_ir->get_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        const auto& shape_infer_consumers = ov::snippets::utils::get_first_child_shape_infer_expr_seq(param);
        const auto& out = shape_infer_consumers.empty() ? param->get_output_port(0)
                                                        : shape_infer_consumers.back()->get_output_port(0);
        const auto consumers = out.get_connected_ports();

        for (const auto& consumer : consumers) {
            auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(consumer.get_expr()->get_node());
            if (brgemm && brgemm_utils::with_repacking(brgemm->get_type()) && consumer.get_index() == 1) {
                const auto src_prc = brgemm->get_input_element_type(0);
                const auto wei_prc = brgemm->get_input_element_type(1);
                const auto isa = brgemm_utils::get_primitive_isa(src_prc, brgemm_utils::with_amx(brgemm->get_type()));
                const auto inner_n_block = brgemm_utils::repacking::compute_inner_n_block(wei_prc);
                const auto is_transposed_b =
                    BrgemmCopyB::is_transposed(m_configurator->get_io_descs()[i]->get_layout());
                auto config = BrgemmCopyBKernelConfig(src_prc, wei_prc, isa, false, is_transposed_b, inner_n_block);
                m_executors[i] = std::make_shared<BrgemmCopyBKernelExecutor>(configurator->get_cache(), config);
            }
        }
    }
}

VectorDims BrgemmExternalRepackingAdjuster::get_blk_order(size_t shape_rank) {
    VectorDims order(shape_rank - brgemm_kernel_rank);
    std::iota(order.begin(), order.end(), 0);
    const auto last_idx = shape_rank - 1;
    order.insert(order.end(), {last_idx - 1, last_idx, last_idx - 1});
    return order;
}

VectorDims BrgemmExternalRepackingAdjuster::get_blk_shape(const VectorDims& planar_shape,
                                                          ov::element::Type prc,
                                                          bool is_transposed) {
    const auto K = *++planar_shape.rbegin();
    const auto N = *planar_shape.rbegin();
    const auto buffer_b_shape = brgemm_utils::repacking::compute_buffer_b_allocation_shape(K, N, prc, is_transposed);
    OPENVINO_ASSERT(buffer_b_shape.size() == 3, "Unexpected buffer B shape rank");
    VectorDims blk_shape(planar_shape.begin(), planar_shape.end() - brgemm_kernel_rank);
    blk_shape.insert(blk_shape.end(), buffer_b_shape.cbegin(), buffer_b_shape.cend());
    return blk_shape;
}

void BrgemmExternalRepackingAdjuster::update_kernel(const RepackExecutorPtr& executor,
                                                    const VectorDims& shape,
                                                    const VectorDims& layout,
                                                    size_t N,
                                                    size_t K,
                                                    ov::element::Type prc) {
    const auto generic_config = executor->get_config().get_clone_ptr();
    auto config = static_cast<BrgemmCopyBKernelConfig*>(generic_config.get());
    const auto idx = config->is_transposed_B() ? 0 : 1;
    const auto copy_wei_stride = ov::snippets::utils::get_dim_in_stride(shape, layout, idx) * prc.size();
    const auto LDB = static_cast<int64_t>(brgemm_utils::repacking::compute_repacked_n_dim(N, prc));
    OPENVINO_ASSERT(LDB >= 0, "Invalid LDB value (less than 0)");
    config->update(N, N, K, K, copy_wei_stride, LDB);
    executor->update_by_config(*config);
}

bool BrgemmExternalRepackingAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmExternalRepackingAdjuster")
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());

    size_t data_size = 0;
    for (const auto& p : m_executors) {
        const auto& i = p.first;
        const auto& shape = cpu_config->io_shapes[i];

        const auto& layout = cpu_config->io_layouts[i];
        const auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        const auto& K = *++planar_shape.rbegin();
        const auto& N = *planar_shape.rbegin();

        const auto& prc = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        const auto blk_shape = get_blk_shape(planar_shape, prc, BrgemmCopyB::is_transposed(layout));

        // src data + dst data per kernel call
        const auto src_data = N * K * prc.size();
        const auto dst_data =
            std::accumulate(blk_shape.rbegin(), blk_shape.rbegin() + 3, prc.size(), std::multiplies<>());
        data_size += src_data + dst_data;

        update_kernel(p.second, shape, layout, N, K, prc);
    }

    const auto cache_size = dnnl::utils::get_cache_size(1, true) + dnnl::utils::get_cache_size(2, true);
    const auto fit_into_cache = data_size < cache_size;
    // Heuristic: If external repacking data doesn't fit in the caches L1 and L2,
    //            external repacking should be executed in seperate parallel section before kernel execution.
    cpu_config->repacking_impl_type = fit_into_cache ? CPURuntimeConfig::RepackingImplType::IN_PARALLEL
                                                     : CPURuntimeConfig::RepackingImplType::SEPARATE;

    const auto is_impl_parallel = cpu_config->repacking_impl_type == CPURuntimeConfig::RepackingImplType::IN_PARALLEL;

    for (const auto& p : m_executors) {
        const auto& i = p.first;
        const auto& shape = cpu_config->io_shapes[i];
        const auto& layout = cpu_config->io_layouts[i];
        auto& repacked_in = cpu_config->repacked_inputs[i];

        const auto& prc = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        auto blk_shape = get_blk_shape(planar_shape, prc, BrgemmCopyB::is_transposed(layout));
        // In parallel impl, each thread needs buffer with only shape [K_blk, N_blk, VNNI] to store repacking data
        if (is_impl_parallel) {
            std::fill(planar_shape.rbegin() + brgemm_kernel_rank, planar_shape.rend(), 1);
            std::fill(blk_shape.rbegin() + brgemm_kernel_rank + 1, blk_shape.rend(), 1);
        }
        const auto order = get_blk_order(planar_shape.size());
        const auto desc = std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(planar_shape), blk_shape, order);

        // Save original input offsets for input before repacking.
        // If the shape has not been changed, it means that we already created `RepackedInput` for this input
        // on previous pass call and now `cpu_config->io_data_offsets[i]` contains offsets not for original input -
        // they were updated for blocked shapes/zeroed for previous initialization and we cannot use them as original
        // offsets.
        const auto in_offsets =
            shape == cpu_config->latest_shapes[i] ? repacked_in.in_offsets() : cpu_config->io_data_offsets[i];

        // In parallel case Kernel should not add offsets to repacked inputs because
        // they will be applied during repacking in execution stage
        if (is_impl_parallel) {
            auto& offsets = cpu_config->io_data_offsets[i];
            std::fill(offsets.begin(), offsets.end(), 0);
        } else {
            ov::snippets::VectorDims shape_for_offset(cpu_config->tensor_rank - shape.size(), 1);
            shape_for_offset.insert(shape_for_offset.end(), blk_shape.begin(), blk_shape.end());
            m_configurator->compute_offsets(shape_for_offset, i, 0);
        }
        const auto out_offsets = cpu_config->io_data_offsets[i];

        repacked_in = RepackedInput(p.second->get_kernel(), desc, in_offsets, out_offsets);
    }

    return true;
}

}  // namespace ov::intel_cpu
