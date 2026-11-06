// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/ngram_hash_function.hpp"
#include "openvino/reference/ngram_hash_function.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool NgramHash::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::as_type_ptr<const ov::op::internal::NGramHashFunction>(op)) {
            errorMessage = "Only internal NGramHashFunction operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

NgramHash::NgramHash(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto ngram_hash = ov::as_type_ptr<const ov::op::internal::NGramHashFunction>(op);
    m_ngram_size = static_cast<size_t>(ngram_hash->get_ngram_size());
    m_heads_per_order = ngram_hash->get_heads_per_order();
    m_multipliers = ngram_hash->get_multipliers();
    m_head_vocab_sizes = ngram_hash->get_head_vocab_sizes();
    m_head_offsets = ngram_hash->get_head_offsets();
}

void NgramHash::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    m_precision = getOriginalInputPrecisionAtPort(0);
    if (none_of(m_precision, ov::element::i32, ov::element::i64)) {
        m_precision = ov::element::i32;
    }
    addSupportedPrimDesc({{LayoutType::ncsp, m_precision}},
                         {{LayoutType::ncsp, ov::element::i32}},
                         impl_desc_type::ref_any);
}

template <typename T>
void NgramHash::executeImpl() {
    const auto* ngrams = getSrcDataAtPortAs<const T>(0);
    auto* row_ids = getDstDataAtPortAs<int32_t>(0);
    const size_t num_tokens = ov::shape_size(getSrcMemoryAtPort(0)->getStaticDims()) / m_ngram_size;
    const size_t num_heads = m_head_vocab_sizes.size();

    // The reference kernel (plain % modulo) is enough here: the Gather from the n-gram embedding table that follows
    // the hash dominates the n-gram subgraph, so the hash is a marginal part of it even when its own cost grows.
    auto hash_range = [&](size_t begin, size_t end) {
        ov::reference::ngram_hash_function(ngrams,
                                           row_ids,
                                           begin,
                                           end,
                                           m_ngram_size,
                                           m_heads_per_order.data(),
                                           m_multipliers.data(),
                                           m_head_vocab_sizes.data(),
                                           m_head_offsets.data(),
                                           num_heads);
    };

    constexpr size_t serial_window_limit = 256;
    if (num_tokens <= serial_window_limit) {
        hash_range(0, num_tokens);
        return;
    }
    const auto num_threads = static_cast<int>(
        std::min(static_cast<size_t>(context->getCpuParallel()->get_num_worker_threads()), num_tokens));
    parallel_nt_static(num_threads, [&](const int ithr, const int nthr) {
        size_t begin = 0;
        size_t end = 0;
        splitter(num_tokens, nthr, ithr, begin, end);
        hash_range(begin, end);
    });
}

namespace {
struct NgramHashContext {
    NgramHash& node;
};
}  // namespace

template <typename T>
struct NgramHash::NgramHashExecute {
    void operator()(NgramHashContext& ctx) {
        ctx.node.executeImpl<T>();
    }
};

void NgramHash::execute([[maybe_unused]] const dnnl::stream& strm) {
    NgramHashContext ctx = {*this};
    OV_SWITCH(intel_cpu,
              NgramHashExecute,
              ctx,
              m_precision,
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::i64, int64_t))
}

void NgramHash::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool NgramHash::created() const {
    return getType() == Type::NgramHash;
}

}  // namespace ov::intel_cpu::node
