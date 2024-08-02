// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "causal_mask_preprocess.h"

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

/*
CausalMaskPreprocess:
    inputs:
        0: attention_mask            : i64[N, kv_len]
                                    0 means mask-out, 1 means attends to
        1: batch_size (size_Gather)  : i32[1]
        2: cache_positions  i32[q_len];
        3: kvLen            i32[1];
    outputs
        0: causal mask for SDPA : f32[batch_size, 1, q_len, kvLen]

The functionality is equivalent to following python code:

    ##### preprocess
    min_dtype = torch.finfo(dtype).min
    causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype
    causal_mask = causal_mask.to(dtype=dtype, device=device)

    mask_length = attention_mask.shape[-1]
    padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
    causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

    ##### when being used will be further sliced
    causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : key_states.shape[-2]]
*/
template <typename T>
struct CausalMaskPreprocess::ExecutorCausalMaskPreprocess : public CausalMaskPreprocess::Executor {
    void execute(dnnl::stream strm,
                 intel_cpu::Node * pnode,
                 const intel_cpu::CausalMaskPreprocessNode::Config& config) override {
        ov::intel_cpu::PlainTensor t_attention_mask(pnode->getSrcMemoryAtPort(0));
        ov::intel_cpu::PlainTensor t_batch_size(pnode->getSrcMemoryAtPort(1));
        ov::intel_cpu::PlainTensor t_cache_positions(pnode->getSrcMemoryAtPort(2));
        ov::intel_cpu::PlainTensor t_kvLen(pnode->getSrcMemoryAtPort(3));

        auto mask_length = t_attention_mask.size(-1);
        auto batch_size = static_cast<size_t>(*t_batch_size.ptr<int32_t>(0));
        auto kvLen = static_cast<size_t>(*t_kvLen.ptr<int32_t>(0));
        auto qLen = t_cache_positions.size(0);

        VectorDims newDims{batch_size, 1, qLen, kvLen};
        pnode->redefineOutputMemory({newDims});
        ov::intel_cpu::PlainTensor t_dst(pnode->getDstMemoryAtPort(0));

        DEBUG_LOG("CausalMaskPreprocess::execute", config.type, "  batch_size=", batch_size, " qLen=", qLen, " kvLen=", kvLen);
        DEBUG_LOG("CausalMaskPreprocess::execute  attention_mask=", t_attention_mask);
        DEBUG_LOG("CausalMaskPreprocess::execute  cache_positions=", t_cache_positions);

        // raw_causal_mask is already ensured to be triu by transformation
        auto* prow = t_cache_positions.ptr<int32_t>(0);
        T min_dtype = std::numeric_limits<T>::lowest();

        parallel_for2d(batch_size, qLen, [&](size_t n, size_t i) {
            auto* pamask = t_attention_mask.ptr<int32_t>(n, 0);
            auto* pdst = t_dst.ptr<T>(n, 0, i);
            auto row = static_cast<size_t>(prow[i]);
            size_t j = 0;
            for (; j < mask_length; j++) {
                bool cmask_eq0 = (j <= row);
                bool amask_eq0 = (pamask[j] == 0);
                bool padding_mask = (cmask_eq0 && amask_eq0);
                pdst[j] = (padding_mask | (!cmask_eq0))? min_dtype : T(0);
            }
            for (; j < kvLen; j++) {
                bool cmask_eq0 = (j <= row);
                pdst[j] = cmask_eq0 ? T(0) : min_dtype;
            }
        });
        DEBUG_LOG("CausalMaskPreprocess::execute  dst=", t_dst);
    }
};

CausalMaskPreprocess::CausalMaskPreprocess(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const intel_cpu::CausalMaskPreprocessNode>(op);
    m_config = node->get_config();
}

bool CausalMaskPreprocess::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const intel_cpu::CausalMaskPreprocessNode>(op);
        if (!node) {
            errorMessage = "Only CausalMaskPreprocessNode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void CausalMaskPreprocess::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<ov::element::Type> iprecs = getOriginalInputPrecisions();
    std::vector<ov::element::Type> oprecs = getOriginalOutputPrecisions();

    // precision preferences
    if (m_config.type == "CausalMaskPreprocess") {
        if (oprecs[0] == ov::element::bf16) {
            m_executor = std::make_shared<ExecutorCausalMaskPreprocess<ov::bfloat16>>();
        } else {
            // fallback to default precision
            m_executor = std::make_shared<ExecutorCausalMaskPreprocess<float>>();
            oprecs[0] = ov::element::f32;
        }
        // all input precisions must be int32
        for (auto& prec : iprecs) prec = ov::element::i32;
    } else {
        OPENVINO_THROW("CPU: CausalMaskPreprocess type not supported : " + m_config.type);
    }

    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++)
        inPortConfigs.emplace_back(LayoutType::ncsp, iprecs[i], getInputShapeAtPort(i), false, -1);

    std::vector<PortConfigurator> outPortConfigs;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++)
        outPortConfigs.emplace_back(LayoutType::ncsp, oprecs[i], getOutputShapeAtPort(i), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void CausalMaskPreprocess::execute(dnnl::stream strm) {
    m_executor->execute(strm, this, m_config);
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
