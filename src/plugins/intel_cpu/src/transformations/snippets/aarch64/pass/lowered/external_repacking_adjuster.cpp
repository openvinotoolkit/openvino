// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "external_repacking_adjuster.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

#include "cpu_shape.h"
#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"

namespace ov::intel_cpu::pass::aarch64 {
namespace {

ov::snippets::VectorDims get_allocation_shape(const ov::snippets::VectorDims& planar_shape,
                                              const ov::element::Type& precision) {
    OPENVINO_ASSERT(planar_shape.size() >= 2, "GEMM weights must have rank >= 2");

    const auto K = *++planar_shape.rbegin();
    const auto N = *planar_shape.rbegin();
    OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(N) && !ov::snippets::utils::is_dynamic_value(K),
                    "N and K shape should not be dynamic for repacked aarch64 GEMM weights");

    const auto packed_bytes = ov::intel_cpu::aarch64::gemm_utils::repacking::get_rhs_packed_size(precision, N, K);
    OPENVINO_ASSERT(packed_bytes % precision.size() == 0, "Unexpected packed weights byte size alignment");

    auto allocation_shape = planar_shape;
    allocation_shape[allocation_shape.size() - 2] = 1;
    allocation_shape[allocation_shape.size() - 1] = packed_bytes / precision.size();
    return allocation_shape;
}

ov::snippets::VectorDims get_repacked_offsets(const ov::snippets::VectorDims& planar_shape,
                                              size_t target_rank,
                                              const ov::element::Type& precision) {
    const auto allocation_shape = get_allocation_shape(planar_shape, precision);
    OPENVINO_ASSERT(target_rank >= allocation_shape.size(), "Incorrect target rank for repacked GEMM weights offsets");
    ov::snippets::VectorDims shape_for_offset(target_rank - allocation_shape.size(), 1);
    shape_for_offset.insert(shape_for_offset.end(), allocation_shape.begin(), allocation_shape.end());

    ov::snippets::VectorDims dst_offsets;
    ov::snippets::utils::init_strides(shape_for_offset, target_rank, precision.size(), 0, dst_offsets);
    return dst_offsets;
}

}  // namespace

const size_t GemmExternalRepackingAdjuster::gemm_kernel_rank = 2;

GemmExternalRepackingAdjuster::GemmExternalRepackingAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                             const CPURuntimeConfigurator* configurator)
    : ov::snippets::lowered::pass::RuntimeOptimizer(configurator) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());
    const auto& input_repackers = cpu_config->input_repackers;
    const auto& params = linear_ir->get_parameters();
    for (const auto& [idx, input_repacker] : input_repackers) {
        OPENVINO_ASSERT(idx < params.size(), "Incorrect index of repacked input");

        auto config = RepackedInputConfig{};
        if (!input_repacker.already_repacked()) {
            config.needs_runtime_repacking = true;
        }
        m_repacked_inputs.emplace(idx, config);
    }
    OPENVINO_ASSERT(input_repackers.size() == m_repacked_inputs.size(), "Incorrect count of repacked inputs");
}

CpuBlockedMemoryDescPtr GemmExternalRepackingAdjuster::get_desc(const ov::snippets::VectorDims& planar_shape,
                                                                const ov::element::Type& prc) {
    const auto allocation_shape = get_allocation_shape(planar_shape, prc);
    const auto blocked_order = ov::snippets::utils::get_planar_layout(allocation_shape.size());
    return std::make_shared<CpuBlockedMemoryDesc>(prc, Shape(planar_shape), allocation_shape, blocked_order);
}

void GemmExternalRepackingAdjuster::update_kernel(const RepackExecutorPtr& executor,
                                                  const ov::snippets::VectorDims& shape,
                                                  const ov::snippets::VectorDims& layout,
                                                  size_t N,
                                                  size_t K,
                                                  const ov::element::Type& prc) {
    OPENVINO_ASSERT(executor, "Input marked for runtime repacking has no executor");
    OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(N) && !ov::snippets::utils::is_dynamic_value(K),
                    "N and K shape should not be dynamic at GemmExternalRepackingAdjuster update kernel stage.");

    const auto row_stride_bytes = ov::snippets::utils::get_dim_in_stride(shape, layout, 1) * prc.size();
    const auto col_stride_bytes = ov::snippets::utils::get_dim_in_stride(shape, layout, 0) * prc.size();
    const auto is_transposed = ov::snippets::utils::get_input_dim_idx(layout, 0) != layout.size() - 1;

    auto config = executor->get_config();
    config.update(N, K, row_stride_bytes, col_stride_bytes, is_transposed);
    executor->update_by_config(config);
}

GemmExternalRepackingAdjuster::RepackExecutorPtr GemmExternalRepackingAdjuster::create_executor(
    const ov::element::Type& prc) {
    return std::make_shared<ov::intel_cpu::aarch64::GemmCopyBKernel>(prc);
}

bool GemmExternalRepackingAdjuster::run(const snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::GemmExternalRepackingAdjuster")
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_configurator->get_config());

    size_t data_size = 0;
    bool has_runtime_repacking = false;
    for (const auto& [i, repacked_input] : m_repacked_inputs) {
        if (!repacked_input.needs_runtime_repacking) {
            continue;
        }
        has_runtime_repacking = true;

        const auto& shape = cpu_config->io_shapes[i];
        const auto& layout = cpu_config->io_layouts[i];
        const auto& prc = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        const auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        const auto& K = *++planar_shape.rbegin();
        const auto& N = *planar_shape.rbegin();

        const auto packed_bytes = ov::intel_cpu::aarch64::gemm_utils::repacking::get_rhs_packed_size(prc, N, K);
        const auto src_data = N * K * prc.size();
        data_size += src_data + packed_bytes;
    }

    if (!has_runtime_repacking) {
        cpu_config->repacking_impl_type = CPURuntimeConfig::RepackingImplType::NONE;
    } else {
        const auto cache_size = dnnl::utils::get_cache_size(1, true) + dnnl::utils::get_cache_size(2, true);
        const auto fit_into_cache = data_size < cache_size;
        // Heuristic: If external repacking data doesn't fit in the caches L1 and L2,
        //            external repacking should be executed in separate parallel section before kernel execution.
        cpu_config->repacking_impl_type = fit_into_cache ? CPURuntimeConfig::RepackingImplType::IN_PARALLEL
                                                         : CPURuntimeConfig::RepackingImplType::SEPARATE;
    }

    const auto is_impl_parallel = cpu_config->repacking_impl_type == CPURuntimeConfig::RepackingImplType::IN_PARALLEL;

    for (const auto& [i, repacked_input] : m_repacked_inputs) {
        if (!repacked_input.needs_runtime_repacking) {
            continue;
        }

        const auto& shape = cpu_config->io_shapes[i];
        const auto& layout = cpu_config->io_layouts[i];
        auto& input_repacker = cpu_config->input_repackers[i];

        const auto& prc = linear_ir.get_parameters()[i]->get_node()->get_output_element_type(0);
        auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);
        const auto& K = *++planar_shape.rbegin();
        const auto& N = *planar_shape.rbegin();
        const auto executor = create_executor(prc);
        update_kernel(executor, shape, layout, N, K, prc);

        // In parallel impl, each thread needs buffer with only inner shape to store repacking data.
        if (is_impl_parallel) {
            const auto batch_count = planar_shape.size() - gemm_kernel_rank;
            std::fill(planar_shape.begin(), planar_shape.begin() + batch_count, 1);
        }

        const auto dst_desc = get_desc(planar_shape, prc);

        // Save original input offsets for input before repacking.
        // If the shape has not been changed, it means that we already created `InputRepacker` for this input
        // on previous pass call and now `cpu_config->io_data_offsets[i]` contains offsets not for original input -
        // they were updated for packed shapes/zeroed for previous initialization and we cannot use them as original
        // offsets.
        const auto in_offsets =
            shape == cpu_config->latest_shapes[i] ? input_repacker.in_offsets() : cpu_config->io_data_offsets[i];

        // In parallel case Kernel should not add offsets to repacked inputs because
        // they will be applied during repacking in execution stage.
        if (is_impl_parallel) {
            auto& offsets = cpu_config->io_data_offsets[i];
            std::fill(offsets.begin(), offsets.end(), 0);
        } else {
            cpu_config->io_data_offsets[i] = get_repacked_offsets(planar_shape, in_offsets.size(), prc);
        }
        const auto out_offsets = cpu_config->io_data_offsets[i];

        input_repacker = InputRepacker(executor, dst_desc, in_offsets, out_offsets);
    }

    // Inputs prepacked during model preparation do not need runtime repacking kernels.
    // For them, we only adjust packed output offsets for current runtime shapes.
    for (const auto& [idx, repacked_input] : m_repacked_inputs) {
        if (repacked_input.needs_runtime_repacking) {
            continue;
        }
        const auto& shape = cpu_config->io_shapes[idx];
        const auto& layout = cpu_config->io_layouts[idx];
        const auto& precision = linear_ir.get_parameters()[idx]->get_node()->get_output_element_type(0);
        const auto planar_shape = ov::snippets::utils::get_planar_vdims(shape, layout);

        cpu_config->io_data_offsets[idx] =
            get_repacked_offsets(planar_shape, cpu_config->io_data_offsets[idx].size(), precision);
        cpu_config->input_repackers.erase(idx);
    }
    return true;
}

}  // namespace ov::intel_cpu::pass::aarch64
