// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

namespace ov::intel_cpu::node {

namespace qsa {
class BrgemmBlockScorer;
}  // namespace qsa

/// @brief F32 reference implementation of ov::intel_cpu::QSAIndexer (padding-aware,
/// non-paged, caller-owned-cache block-level indexer for Qwen Sparse Attention).
///
/// Unlike QSAIndexer, this op is fed a per-row `key_length` (total valid length, past + current,
/// after this call) instead of a `past_lens` running counter: `past`/`current`/block-append-offset
/// are all derived per row from `key_length` and this call's window length alone, so divergent
/// per-row batches and intra-call left-padding are handled without any extra persisted state.
/// `past_incomplete_raw_k` / `past_incomplete_rope` are FIXED-size (not growing) caller-owned
/// state tensors holding the current incomplete block's raw rows / reference RoPE coefficients;
/// they are fully rewritten every call rather than grown. The three state inputs
/// (`summary_cache`, `past_incomplete_raw_k`, `past_incomplete_rope`) are never modified: the
/// updated states are written to dedicated outputs for an outside Assign to consume. `summary_cache` grows like
/// QSAPrepare-17's `block_index_keys`: the output has max(key_length) / key_block_size blocks, so the output shape
/// depends on the `key_length` values. All floating-point ports (inputs and state outputs) share one
/// precision (f32, f16 or bf16, that of `q_idx`); compute is always performed in fp32.
/// `sel_indices`/`sel_count` are always i32.
class QSAIndexer : public Node {
    using InputIdx = ov::intel_cpu::QSAIndexer::InputIdx;

public:
    QSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::QSAIndexer;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    // summary_cache is empty on the first call (no earlier blocks), the node still has to run then.
    [[nodiscard]] bool isExecutable() const override {
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            if (i != InputIdx::SUMMARY_CACHE && isInputTensorAtPortEmpty(i)) {
                return false;
            }
        }
        return true;
    }
    // The summary_cache output shape depends on the key_length values, not only on the input shapes.
    [[nodiscard]] bool needShapeInfer() const override {
        return true;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename T>
    void execute_typed();

    ov::intel_cpu::QSAIndexer::Config m_config{};
    // The scorer of the block scoring on brgemm (see qsa_brgemm_scorer.hpp), taken from the params cache of the graph
    // context in createPrimitive(): the layers of the same configuration share it, and nothing is built while the model
    // runs. Null when this CPU or configuration has no brgemm scoring (the scalar scoring runs then).
    std::shared_ptr<const qsa::BrgemmBlockScorer> m_scorer;
};

}  // namespace ov::intel_cpu::node
