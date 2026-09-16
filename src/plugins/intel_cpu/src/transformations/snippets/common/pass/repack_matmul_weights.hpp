// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "emitters/snippets/input_repacker.hpp"
#include "graph_context.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface RepackMatMulWeights
 * @brief Repack constant MatMul weights to the target backend format.
 * @ingroup snippets
 */
class RepackMatMulWeights : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RepackMatMulWeights");
    RepackMatMulWeights(GraphContext::CPtr context,
                        ov::intel_cpu::InputRepackerMap& input_repackers,
                        std::vector<MemoryPtr>& src_mem_ptrs,
                        std::set<size_t> compile_time_repacking_idxs)
        : m_context(std::move(context)),
          m_input_repackers(input_repackers),
          m_src_mem_ptrs(src_mem_ptrs),
          m_compile_time_repacking_idxs(std::move(compile_time_repacking_idxs)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

protected:
    struct MatMulWeightsSource {
        VectorDims shape;
        VectorDims layout;
    };

    struct RepackedMatMulWeights {
        MemoryPtr memory;
        CpuBlockedMemoryDescPtr desc;
    };

    [[nodiscard]] static MatMulWeightsSource get_weights_source(const std::shared_ptr<ov::Node>& matmul_node,
                                                                const MemoryPtr& orig_src_mem_ptr);
    [[nodiscard]] static CpuBlockedMemoryDescPtr get_src_cpu_desc(const MatMulWeightsSource& source,
                                                                  ov::element::Type precision);

    /**
     * @brief Repack a constant MatMul weights input for a backend-specific MatMul/GEMM consumer.
     * @param consumer Backend MatMul/GEMM node that consumes the constant weights.
     * @param source Logical weights shape and layout after CopyB extraction. When extraction inserts a graph Reorder,
     * this metadata is not recoverable from @p orig_src_mem_ptr alone.
     * @param orig_src_mem_ptr Original constant weights memory buffer.
     * @return Repacked weights memory with its CPU descriptor.
     */
    [[nodiscard]] virtual RepackedMatMulWeights repack(const std::shared_ptr<ov::Node>& consumer,
                                                       const MatMulWeightsSource& source,
                                                       const MemoryPtr& orig_src_mem_ptr) = 0;

    const GraphContext::CPtr m_context;
    ov::intel_cpu::InputRepackerMap& m_input_repackers;
    std::vector<MemoryPtr>& m_src_mem_ptrs;
    const std::set<size_t> m_compile_time_repacking_idxs;
};

}  // namespace ov::intel_cpu::pass
