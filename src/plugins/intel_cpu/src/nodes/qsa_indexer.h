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
#include "nodes/kernels/sparse_attn/qsa_executor.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/node.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

namespace ov::intel_cpu::node {

/// @brief CPU implementation of ov::intel_cpu::QSAIndexerNode: the non-paged block-level indexer of Qwen Sparse
/// Attention with caller-owned caches.
///
/// The op is fed a per-row `key_length` (total valid length, past + current, after this call): `past`, `current` and
/// the block-append offset are derived per row from it and this call's window length, so divergent rows and left
/// padding need no extra state. `past_incomplete_raw_k` / `past_incomplete_rope` are fixed-size states holding the
/// current incomplete block, fully rewritten every call. The three state inputs (`summary_cache`,
/// `past_incomplete_raw_k`, `past_incomplete_rope`) are never modified: the updated states go to dedicated outputs for
/// an outside Assign. `summary_cache` has max(key_length) / key_block_size blocks, so its output shape depends on the
/// `key_length` values. All floating-point ports share one precision (f32, f16 or bf16, that of `q_idx`); compute is
/// in f32. `sel_indices`/`sel_count` are i32.
class QSAIndexer : public Node {
    using InputIdx = ov::intel_cpu::QSAIndexerNode::InputIdx;

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
    // The computation is the cross-compiled executor (AVX-512 / AVX2 / plain C++ by the CPU); the scoring of the blocks
    // runs on the brgemm scorer instead where the CPU, the precision and the head counts allow it.
    ov::intel_cpu::QSAIndexerNode::Config m_config;
    size_t m_index_query_heads = 0;  // fixed by the model
    size_t m_index_key_heads = 0;
    std::shared_ptr<ov::Extensions::Cpu::qsa::DenseExecutor> m_executor;
    std::shared_ptr<ov::Extensions::Cpu::qsa::DenseScorer> m_scorer;  // null: the executor scores the blocks itself
    impl_desc_type m_impl = impl_desc_type::ref;
};

}  // namespace ov::intel_cpu::node
