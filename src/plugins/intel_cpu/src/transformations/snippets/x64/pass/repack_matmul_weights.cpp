// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_matmul_weights.hpp"

#include <cstddef>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reorder.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass {

using namespace brgemm_utils;

DnnlMemoryDescPtr RepackMatMulWeights::get_src_desc(const VectorDims& shape,
                                                    const VectorDims& layout,
                                                    const BrgemmConfig& brgemm_config) {
    const auto planar_shape = Shape{ov::snippets::utils::get_preordered_vdims(shape, layout)};
    return MemoryDescUtils::convertToDnnlMemoryDesc(
        std::make_shared<CpuBlockedMemoryDesc>(brgemm_config.orig_wei_dt(), planar_shape, shape, layout));
}

DnnlMemoryDescPtr RepackMatMulWeights::get_dst_desc(const Shape& shape, const BrgemmConfig& brgemm_config) {
    const auto [blocked_dims, blocked_order] =
        brgemm_utils::repacking::get_wei_blocked_shape(shape.getStaticDims(),
                                                       brgemm_config.wei_dt(),
                                                       brgemm_config.wei_k_blk(),
                                                       brgemm_config.wei_n_blk(),
                                                       brgemm_config.are_wei_blocked());

    return MemoryDescUtils::convertToDnnlMemoryDesc(
        std::make_shared<CpuBlockedMemoryDesc>(brgemm_config.wei_dt(), shape, blocked_dims, blocked_order));
}

bool RepackMatMulWeights::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RepackMatMulWeights);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::RepackMatMulWeights")

    const auto& params = model->get_parameters();
    std::unordered_set<size_t> weights_idxs;
    for (const auto& [i, input_repacker] : m_input_repackers) {
        const auto& parameter = params[i];

        const auto shape_infer_leaf = ov::snippets::utils::get_leaf_node_of_first_child_shape_infer_seq(parameter);
        const auto& first_child = shape_infer_leaf ? shape_infer_leaf : parameter;
        const auto consumers = first_child->output(0).get_target_inputs();

        const auto brgemm_cpu = ov::as_type_ptr<BrgemmCPU>(consumers.cbegin()->get_node()->shared_from_this());
        OPENVINO_ASSERT(consumers.size() == 1 && brgemm_cpu != nullptr, "Expected one consumer - BrgemmCPU");

        const auto& brgemm_config = brgemm_cpu->get_config();
        if (!brgemm_config.are_wei_constant()) {
            continue;
        }

        const auto& orig_src_mem_ptr = m_src_mem_ptrs[i];

        VectorDims shape;
        VectorDims layout;
        if (const auto& reorder =
                ov::as_type_ptr<ov::snippets::op::Reorder>(brgemm_cpu->input_value(1).get_node_shared_ptr())) {
            const auto& port_desc =
                ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(reorder->input(0));
            layout = port_desc->get_layout();
            shape = ov::snippets::utils::pshape_to_vdims(reorder->get_input_partial_shape(0));
        } else {
            shape = orig_src_mem_ptr->getShape().getStaticDims();
            layout.resize(shape.size());
            std::iota(layout.begin(), layout.end(), 0);
        }

        const auto& eng = m_context->getEngine();
        const auto src_mem_desc = get_src_desc(shape, layout, brgemm_config);
        const auto dst_mem_desc = get_dst_desc(src_mem_desc->getShape(), brgemm_config);
        const auto src_mem_ptr = std::make_shared<Memory>(eng, src_mem_desc, orig_src_mem_ptr->getData());

        // Pass empty privateWeightCache because:
        //  - there might be several FCs in Subgraph. Then each FC should have own private cache
        //  - currently, we repack weights only once on model compilation stage - no need to save them to the private
        //    cache
        m_src_mem_ptrs[i] = ov::intel_cpu::utils::prepareWeightsMemory(src_mem_desc,
                                                                       dst_mem_desc,
                                                                       src_mem_ptr,
                                                                       eng,
                                                                       m_context->getParamsCache(),
                                                                       m_context->getWeightsCache(),
                                                                       nullptr);
        weights_idxs.insert(i);
    }

    // Removed already repacked inputs: remaining inputs will be repacked in runtime configurator on inference stage
    for (const auto& weight_idx : weights_idxs) {
        m_input_repackers.erase(weight_idx);
    }

    return !weights_idxs.empty();
}
}  // namespace ov::intel_cpu::pass
