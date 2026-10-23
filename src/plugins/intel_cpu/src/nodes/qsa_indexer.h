// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "transformations/cpu_opset/x64/op/qsa_indexer.hpp"

namespace ov::intel_cpu::node {

/// @brief F32 reference implementation of ov::intel_cpu::QSAIndexer (padding-aware,
/// non-paged, caller-owned-cache block-level indexer for Qwen Sparse Attention).
///
/// Unlike QSAIndexer, this op is fed a per-row `key_length` (total valid length, past + current,
/// after this call) instead of a `past_lens` running counter: `past`/`current`/block-append-offset
/// are all derived per row from `key_length` and this call's window length alone, so divergent
/// per-row batches and intra-call left-padding are handled without any extra persisted state.
/// `past_incomplete_raw_k` / `past_incomplete_rope` are FIXED-size (not growing) caller-owned
/// writable tensors holding the current incomplete block's raw rows / reference RoPE
/// coefficients; they are fully overwritten every call rather than grown. All floating-point data
/// ports may be negotiated as f32, f16, or bf16 independently; compute is always performed in
/// fp32. The two outputs (`sel_indices`, `sel_count`) are always i32.
class QSAIndexer : public Node {
public:
    QSAIndexer(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {}
    [[nodiscard]] bool created() const override {
        return getType() == Type::QSAIndexer;
    }
    [[nodiscard]] bool needPrepareParams() const override {
        return false;
    }
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    void createPrimitive() override {}
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    template <typename IndexT>
    void execute_typed();

    ov::intel_cpu::QSAIndexer::Config m_config{};
};

}  // namespace ov::intel_cpu::node
