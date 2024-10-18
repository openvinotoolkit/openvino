// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attn.h"

#include "common/arbitrary_order_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/util/common_util.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"

#include "utils/plain_tensor.hpp"
#include "kernels/scaled_attn/executor_pa.hpp"
#include "kernels/scaled_attn/attn_memcpy.hpp"
#include "kernels/scaled_attn/attn_quant.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace ov::Extensions::Cpu;
using namespace ov::Extensions::Cpu::XARCH;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

struct PagedAttentionKey {
    ov::element::Type rtPrecision;

    size_t hash() const;
    bool operator==(const PagedAttentionKey& rhs) const;
};

size_t PagedAttentionKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, rtPrecision.hash());

    return seed;
}

bool PagedAttentionKey::operator==(const PagedAttentionKey& rhs) const {
    auto retVal = rtPrecision == rhs.rtPrecision;

    return retVal;
}

PagedAttention::PagedAttention(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    // output score may have no child
    m_hasScore = !op->get_output_target_inputs(1).empty();
}

void PagedAttention::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto rtPrecision = getRuntimePrecision();

    NodeConfig config;
    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto orgInputNumber = getOriginalInputsNumber();
    config.inConfs.resize(orgInputNumber);
    config.outConfs.resize(getOriginalOutputsNumber());
    config.inConfs[PagedAttentionExecutor::ID_Q].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(PagedAttentionExecutor::ID_Q)));
    config.inConfs[PagedAttentionExecutor::ID_K].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(PagedAttentionExecutor::ID_K)));
    config.inConfs[PagedAttentionExecutor::ID_V].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getInputShapeAtPort(PagedAttentionExecutor::ID_V)));

    OPENVINO_ASSERT(orgInputNumber == 13, "The input number of PagedAttention should be 13.");
    // kvcache, float, []
    auto past_kv_input_mem_precision = getOriginalInputPrecisionAtPort(PagedAttentionExecutor::ID_KCACHE);
    config.inConfs[PagedAttentionExecutor::ID_KCACHE].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        past_kv_input_mem_precision, getInputShapeAtPort(PagedAttentionExecutor::ID_KCACHE)));
    config.inConfs[PagedAttentionExecutor::ID_VCACHE].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        past_kv_input_mem_precision, getInputShapeAtPort(PagedAttentionExecutor::ID_VCACHE)));
    // past_lens, int, [b_seq]
    config.inConfs[PagedAttentionExecutor::ID_PAST_LENS].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(PagedAttentionExecutor::ID_PAST_LENS)));
    // subsequence_begins, int, [b_seq]
    config.inConfs[PagedAttentionExecutor::ID_SUBSEQUENCE_BEGINS].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(PagedAttentionExecutor::ID_SUBSEQUENCE_BEGINS)));
    // block_indices, int, [num_blocks]
    config.inConfs[PagedAttentionExecutor::ID_BLOCK_INDICES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(PagedAttentionExecutor::ID_BLOCK_INDICES)));
    // block_indices_begins, int, [b_seq]
    config.inConfs[PagedAttentionExecutor::ID_BLOCK_INDICES_BEGINS].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(PagedAttentionExecutor::ID_BLOCK_INDICES_BEGINS)));
    // scale, float, []
    config.inConfs[PagedAttentionExecutor::ID_SCALE].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::f32, getInputShapeAtPort(PagedAttentionExecutor::ID_SCALE)));
    // sliding_window, int, []
    config.inConfs[PagedAttentionExecutor::ID_SLIDING_WINDOW].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(PagedAttentionExecutor::ID_SLIDING_WINDOW)));
    // alibi_slopes, float, [H|0]
    config.inConfs[PagedAttentionExecutor::ID_ALIBI_SLOPES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::f32, getInputShapeAtPort(PagedAttentionExecutor::ID_ALIBI_SLOPES)));
    // max_context_len, int, []
    config.inConfs[PagedAttentionExecutor::ID_MAX_CONTEXT_LEN].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::i32, getInputShapeAtPort(PagedAttentionExecutor::ID_MAX_CONTEXT_LEN)));

    config.outConfs[0].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        rtPrecision, getOutputShapeAtPort(0)));
    config.outConfs[1].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(
        ov::element::f32, getOutputShapeAtPort(1)));

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref_any);
}

void PagedAttention::createPrimitive() {
    auto rtPrecision = getRuntimePrecision();

    // in one model, kvCachePrecision could not be changed so no need to care whether it may be changed.
    PagedAttentionKey key = {rtPrecision};

    auto builder = [&](const PagedAttentionKey& key) -> std::shared_ptr<PagedAttentionExecutor> {
#ifdef OPENVINO_ARCH_X86_64
        auto kvCachePrecision = getOriginalInputPrecisionAtPort(PagedAttentionExecutor::ID_KCACHE);
        return make_pa_executor(rtPrecision, kvCachePrecision);
#else
        return nullptr;
#endif
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    if (!result.first) {
        OPENVINO_THROW("PagedAttention AttentionExecutor creation fails with precision " + rtPrecision.to_string());
    }
    m_executor = result.first;
}

void PagedAttention::execute(dnnl::stream strm) {
    auto orginInputNumber = getOriginalInputsNumber();
    std::vector<MemoryPtr> inputs(orginInputNumber);
    std::vector<MemoryPtr> outputs(m_hasScore ? 2 : 1);

    for (size_t i = 0; i < orginInputNumber; i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    auto outDims = inputs[0]->getStaticDims();
    const auto& keyDims = inputs[1]->getStaticDims();
    const auto& valueDims = inputs[2]->getStaticDims();
    // value head_size may be not same with key
    if (keyDims[1] != valueDims[1]) {
        // The outDims[1] should be `num_heads * v_head_size`, it can be got from:
        // because:
        //   q: query_ps[1] = num_heads * head_size
        //   k: key_ps[1] = num_kv_heads * head_size
        //   v: value_ps[1] = num_kv_heads * v_head_size
        // therefore:
        //   q * v / k = (num_heads * head_size) * (num_kv_heads * v_head_size) /
        //               (num_kv_heads * head_size) = num_heads * v_head_size
        outDims[1] = outDims[1] * valueDims[1] / keyDims[1];
    }
    if (m_hasScore) {
        size_t len = 0;
        const auto& pastLensDims = inputs[5]->getStaticDims();
        auto pastLens = inputs[5]->getDataAs<const int32_t>();
        for (size_t i = 0; i < pastLensDims[0]; i++)
            len += pastLens[i];
        len += outDims[0];
        VectorDims scoreDims{len};
        redefineOutputMemory({outDims, scoreDims});
    } else {
        redefineOutputMemory(0, outDims);
    }

    outputs[0] = getDstMemoryAtPort(0);
    if (m_hasScore)
        outputs[1] = getDstMemoryAtPort(1);

    m_executor->execute(inputs, outputs);
}

bool PagedAttention::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        int orgInput = static_cast<int>(op->get_input_size());
        if (op->get_type_name() == std::string("PagedAttentionExtension") && orgInput == PagedAttentionExecutor::ID_SLIDING_WINDOW + 1) {
            return true;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ov::element::Type PagedAttention::getRuntimePrecision() const {
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    // bf16 should be enabled only when platform supports
    if (rtPrecision == ov::element::bf16 && ov::with_cpu_x86_bfloat16()) {
        rtPrecision = ov::element::bf16;
    } else if (rtPrecision == ov::element::f16 && ov::with_cpu_x86_avx512_core_fp16()) {
        rtPrecision = ov::element::f16;
    } else {
        rtPrecision = ov::element::f32;
    }
    return rtPrecision;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
