// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include <oneapi/dnnl/dnnl.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>

#include "cache/multi_cache.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass {

const size_t BrgemmExternalRepackingAdjuster::brgemm_kernel_rank = 2;

BrgemmExternalRepackingAdjuster::BrgemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                                 const CPURuntimeConfigurator* configurator)
    : snippets::lowered::pass::RuntimeOptimizer(configurator) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    const auto& input_repackers = cpu_config->input_repackers;
    const auto& params = linear_ir->get_parameters();
    for (const auto& [idx, _] : input_repackers) {
        OPENVINO_ASSERT(idx < params.size(), "Incorrect index of repacked input");

        m_executors[idx] = create_executor(params[idx], configurator->get_cache());
    }
    OPENVINO_ASSERT(input_repackers.size() == m_executors.size(), "Incorrect count of repacked inputs");
}

BrgemmExternalRepackingAdjuster::RepackExecutorPtr BrgemmExternalRepackingAdjuster::create_executor(
    const ov::snippets::lowered::ExpressionPtr& param,
    const ov::intel_cpu::MultiCacheWeakPtr& cache) {
    RepackExecutorPtr executor = nullptr;

    const auto& shape_infer_consumers = ov::snippets::utils::get_first_child_shape_infer_expr_seq(param);
    const auto& out =
        shape_infer_consumers.empty() ? param->get_output_port(0) : shape_infer_consumers.back()->get_output_port(0);
    const auto consumers = out.get_connected_ports();

    for (const auto& consumer : consumers) {
        auto brgemm = ov::as_type_ptr<ov::intel_cpu::BrgemmCPU>(consumer.get_expr()->get_node());
        if (!brgemm) {
            continue;
        }

        const auto& brgemm_config = brgemm->get_config();
        if (brgemm_config.with_wei_repacking() && consumer.get_index() == 1) {
            OPENVINO_ASSERT(brgemm_config.with_compensations() == false,
                            "External repacking for BrgemmCPU with compensations is not supported.");
            const auto kernel_config = BrgemmCopyBKernelConfig(brgemm_config);
            executor = std::make_shared<BrgemmCopyBKernelExecutor>(cache, kernel_config);
            break;
        }
    }
    OPENVINO_ASSERT(executor, "The executor of the repacked input must be inited!");
    return executor;
}

CpuBlockedMemoryDescPtr BrgemmExternalRepackingAdjuster::get_desc(const ov::snippets::VectorDims& planar_shape,
                                                                  const ov::element::Type& prc,
                                                                  size_t wei_k_blk,
                                                                  size_t wei_n_blk,
                                                                  bool are_wei_blocked,
                                                                  bool is_transposed) {
    const auto allocation_shape = brgemm_utils::repacking::compute_buffer_b_allocation_shape(planar_shape,
                                                                                             prc,
                                                                                             wei_k_blk,
                                                                                             wei_n_blk,
                                                                                             are_wei_blocked,
                                                                                             is_transposed);
    const auto blocked_order = ov::snippets::utils::get_planar_layout(allocation_shape.size());
    return std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(planar_shape), allocation_shape, blocked_order);
}

void BrgemmExternalRepackingAdjuster::update_kernel(const RepackExecutorPtr& executor,
                                                    const VectorDims& shape,
                                                    const VectorDims& layout,
                                                    size_t N,
                                                    size_t K) {
    const auto generic_config = executor->get_config().get_clone_ptr();
    auto* config = static_cast<BrgemmCopyBKernelConfig*>(generic_config.get());
    const auto idx = config->is_transposed_B() ? 0 : 1;
    const auto copy_wei_stride =
        ov::snippets::utils::get_dim_in_stride(shape, layout, idx) * dnnl_data_type_size(config->get_original_wei_dt());
    const auto LDB =
        brgemm_utils::repacking::compute_K_blocked_stride(N, config->get_wei_N_blk(), config->are_wei_blocked());
    config->update(N, N, K, K, copy_wei_stride, LDB);
    executor->update_by_config(*config);
}

bool BrgemmExternalRepackingAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::BrgemmExternalRepackingAdjuster")
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());

    size_t data_size = 0;
    for (const auto& p : m_executors) {
        const auto& i = p.first;
        const auto& executor = p.second;

        const auto& shape = cpu_config->io_shapes[i];
        const auto& layout = cpu_config->io_layouts[i];
        const auto& prc = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        const auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        const auto& K = *++planar_shape.rbegin();
        const auto& N = *planar_shape.rbegin();

        update_kernel(executor, shape, layout, N, K);

        const auto& config = static_cast<const BrgemmCopyBKernelConfig&>(executor->get_config());

        const auto buffer_b_allocation_shape =
            brgemm_utils::repacking::compute_buffer_b_allocation_shape({K, N},
                                                                       prc,
                                                                       config.get_wei_K_blk(),
                                                                       config.get_wei_N_blk(),
                                                                       config.are_wei_blocked(),
                                                                       config.is_transposed_B());

        const auto src_dt_size = dnnl_data_type_size(config.get_original_wei_dt());
        const auto dst_dt_size = dnnl_data_type_size(config.get_wei_dt());
        // src data + dst data per kernel call
        const auto src_data = N * K * src_dt_size;
        const auto dst_data = std::accumulate(buffer_b_allocation_shape.cbegin(),
                                              buffer_b_allocation_shape.cend(),
                                              size_t(1),
                                              std::multiplies<>()) *
                              dst_dt_size;
        data_size += src_data + dst_data;
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
        const auto& executor = p.second;

        const auto& shape = cpu_config->io_shapes[i];
        const auto& layout = cpu_config->io_layouts[i];
        auto& input_repacker = cpu_config->input_repackers[i];

        const auto& prc = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        // In parallel impl, each thread needs buffer with only inner blocked shape to store repacking datata
        if (is_impl_parallel) {
            const auto batch_count = planar_shape.size() - brgemm_kernel_rank;
            std::fill(planar_shape.begin(), planar_shape.begin() + batch_count, 1);
        }

        const auto& config = static_cast<const BrgemmCopyBKernelConfig&>(executor->get_config());
        const auto desc = get_desc(planar_shape,
                                   prc,
                                   config.get_wei_K_blk(),
                                   config.get_wei_N_blk(),
                                   config.are_wei_blocked(),
                                   config.is_transposed_B());

        // Save original input offsets for input before repacking.
        // If the shape has not been changed, it means that we already created `InputRepacker` for this input
        // on previous pass call and now `cpu_config->io_data_offsets[i]` contains offsets not for original input -
        // they were updated for blocked shapes/zeroed for previous initialization and we cannot use them as original
        // offsets.
        const auto in_offsets =
            shape == cpu_config->latest_shapes[i] ? input_repacker.in_offsets() : cpu_config->io_data_offsets[i];

        // In parallel case Kernel should not add offsets to repacked inputs because
        // they will be applied during repacking in execution stage
        if (is_impl_parallel) {
            auto& offsets = cpu_config->io_data_offsets[i];
            std::fill(offsets.begin(), offsets.end(), 0);
        } else {
            const auto blocked_dims = desc->getBlockDims();
            const auto inner_blocks_num = blocked_dims.size() - planar_shape.size();
            const auto rank = in_offsets.size() + inner_blocks_num;  // to align with src offsets rank
            OPENVINO_ASSERT(rank >= blocked_dims.size(), "Incorrect target rank for dst offsets");

            ov::snippets::VectorDims shape_for_offset(rank - blocked_dims.size(), 1);
            shape_for_offset.insert(shape_for_offset.end(), blocked_dims.begin(), blocked_dims.end());

            ov::snippets::VectorDims dst_offsets;
            ov::snippets::utils::init_strides(shape_for_offset, rank, prc.size(), 0, dst_offsets);
            dst_offsets.resize(dst_offsets.size() - inner_blocks_num);
            cpu_config->io_data_offsets[i] = dst_offsets;
        }
        const auto out_offsets = cpu_config->io_data_offsets[i];

        input_repacker = InputRepacker(p.second->get_kernel(), desc, in_offsets, out_offsets);
    }

    return true;
}

}  // namespace ov::intel_cpu::pass
