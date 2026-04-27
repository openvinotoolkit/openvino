// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "emitters/snippets/input_repacker.hpp"
#include "graph_context.h"
#include "openvino/core/model.hpp"
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
                        std::vector<MemoryPtr>& src_mem_ptrs)
        : m_context(std::move(context)),
          m_input_repackers(input_repackers),
          m_src_mem_ptrs(src_mem_ptrs) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    [[maybe_unused]] const GraphContext::CPtr m_context;
    [[maybe_unused]] ov::intel_cpu::InputRepackerMap& m_input_repackers;
    [[maybe_unused]] std::vector<MemoryPtr>& m_src_mem_ptrs;
};

}  // namespace ov::intel_cpu::pass
