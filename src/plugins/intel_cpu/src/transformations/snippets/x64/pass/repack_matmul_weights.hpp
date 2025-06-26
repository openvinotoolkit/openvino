// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <set>
#include <utility>

#include "cpu_memory.h"
#include "emitters/snippets/input_repacker.hpp"
#include "graph_context.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

namespace ov::intel_cpu::pass {

/**
 * @interface RepackMatMulWeights
 * @brief The pass calls plugin-helper "ReorderData" which repacks constant inputs of MatMuls to target blocked format
 * @ingroup snippets
 */
class RepackMatMulWeights : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RepackMatMulWeights");
    RepackMatMulWeights(GraphContext::CPtr context,
                        ov::intel_cpu::InputRepackerMap& input_repackers,
                        std::vector<MemoryPtr>& src_mem_ptrs)
        : m_context(std::move(context)),
          m_input_repackers(input_repackers),
          m_src_mem_ptrs(src_mem_ptrs) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    static DnnlMemoryDescPtr get_src_desc(const VectorDims& shape,
                                          const VectorDims& layout,
                                          const brgemm_utils::BrgemmConfig& brgemm_config);
    static DnnlMemoryDescPtr get_dst_desc(const Shape& shape, const brgemm_utils::BrgemmConfig& brgemm_config);

    const GraphContext::CPtr m_context;
    ov::intel_cpu::InputRepackerMap& m_input_repackers;
    std::vector<MemoryPtr>& m_src_mem_ptrs;
};

}  // namespace ov::intel_cpu::pass
