// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repack_matmul_weights.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/reorder.h"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/itt.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::pass {

using namespace brgemm_utils;

CpuBlockedMemoryDescPtr RepackMatMulWeights::get_src_desc(const VectorDims& orig_shape,
                                                          const BrgemmConfig& brgemm_config) {
    auto weights_2d = reshapeDownToRank<2>(orig_shape);
    auto blocked_dims = weights_2d;
    auto blocked_order = VectorDims{0, 1};
    if (brgemm_config.transposed_b()) {
        weights_2d = {weights_2d[1], weights_2d[0]};
        blocked_order = {1, 0};
    }

    return std::make_shared<CpuBlockedMemoryDesc>(brgemm_config.orig_wei_dt(),
                                                  Shape{weights_2d},
                                                  blocked_dims,
                                                  blocked_order);
}

CpuBlockedMemoryDescPtr RepackMatMulWeights::get_dst_desc(const Shape& weights_2d, const BrgemmConfig& brgemm_config) {
    const auto& K = weights_2d.getStaticDims()[0];
    const auto& N = weights_2d.getStaticDims()[1];
    VectorDims blocked_dims, blocked_order;
    const auto& vnni_factor = compute_vnni_factor(brgemm_config.wei_dt());
    if (brgemm_config.are_wei_blocked()) {
        blocked_dims = {ov::snippets::utils::div_up(N, brgemm_config.wei_n_blk()),
                        ov::snippets::utils::div_up(K, brgemm_config.wei_k_blk() * vnni_factor),
                        brgemm_config.wei_k_blk(),
                        brgemm_config.wei_n_blk(),
                        vnni_factor};
        blocked_order = {1, 0, 0, 1, 0};
    } else {
        blocked_dims = {ov::snippets::utils::div_up(K, vnni_factor),
                        ov::snippets::utils::rnd_up(N, brgemm_config.wei_n_blk()),
                        vnni_factor};
        blocked_order = {0, 1, 0};
    }

    if (vnni_factor == 1) {
        blocked_dims.pop_back();
        blocked_order.pop_back();
    }

    return std::make_shared<CpuBlockedMemoryDesc>(brgemm_config.wei_dt(), weights_2d, blocked_dims, blocked_order);
}

MemoryPtr RepackMatMulWeights::repack(const CpuBlockedMemoryDescPtr& src_desc,
                                      const CpuBlockedMemoryDescPtr& dst_desc,
                                      const MemoryCPtr& src_mem,
                                      const GraphContext::CPtr& context) {
    auto create = [&]() {
        Memory srcMemory{context->getEngine(), src_desc, src_mem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine(), dst_desc);
        node::Reorder::reorderData(srcMemory, *_ptr, context->getParamsCache());
        return _ptr;
    };

    if (auto weight_cache = context->getWeightsCache()) {
        const auto str_hash =
            DnnlExtensionUtils::computeWeightsStringHash(src_mem, MemoryDescUtils::convertToDnnlMemoryDesc(dst_desc));
        return *weight_cache->findOrCreate(str_hash, create);
    } else {
        return create();
    }
}

bool RepackMatMulWeights::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(RepackMatMulWeights);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::RepackMatMulWeights")

    const auto& params = model->get_parameters();
    std::unordered_set<size_t> weights_idxs;
    for (const auto& [i, input_repacker] : m_input_repackers) {
        const auto& parameter = params[i];

        const auto& shape_infer_leaf = ov::snippets::utils::get_leaf_node_of_first_child_shape_infer_seq(parameter);
        const auto& first_child = shape_infer_leaf ? shape_infer_leaf : parameter;
        const auto consumers = first_child->output(0).get_target_inputs();

        const auto brgemm_cpu = ov::as_type_ptr<BrgemmCPU>(consumers.cbegin()->get_node()->shared_from_this());
        OPENVINO_ASSERT(consumers.size() == 1 && brgemm_cpu != nullptr, "Expected one consumer - BrgemmCPU");

        const auto& brgemm_config = brgemm_cpu->get_config();
        if (!brgemm_config.are_wei_constant()) {
            continue;
        }

        const auto& orig_src_mem_ptr = m_src_mem_ptrs[i];
        const auto src_mem_desc = get_src_desc(orig_src_mem_ptr->getStaticDims(), brgemm_config);
        const auto dst_mem_desc = get_dst_desc(src_mem_desc->getShape(), brgemm_config);
        const auto src_mem_ptr =
            std::make_shared<Memory>(m_context->getEngine(), src_mem_desc, orig_src_mem_ptr->getData());

        m_src_mem_ptrs[i] = repack(src_mem_desc, dst_mem_desc, src_mem_ptr, m_context);
        weights_idxs.insert(i);
    }

    for (const auto& weight_idx : weights_idxs) {
        m_input_repackers.erase(weight_idx);
    }

    return !weights_idxs.empty();
}
}  // namespace ov::intel_cpu::pass
