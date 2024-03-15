// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode.h"

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

VNode::VNode(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const intel_cpu::VNode>(op);
    m_config = node->get_config();
}

template <typename T>
struct VNode::VNodeExecutorCausalMaskPreprocess : public VNode::Executor {
    void execute(dnnl::stream strm,
                 const intel_cpu::VNode::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor t_attention_mask(inputs[0]);
        ov::intel_cpu::PlainTensor t_batch_size(inputs[1]);
        ov::intel_cpu::PlainTensor t_cache_positions(inputs[2]);
        ov::intel_cpu::PlainTensor t_kvLen(inputs[3]);
        ov::intel_cpu::PlainTensor t_dst(outputs[0]);

        DEBUG_LOG("VNode::execute  attention_mask=", t_attention_mask);
        DEBUG_LOG("VNode::execute  batch_size=", t_batch_size);
        DEBUG_LOG("VNode::execute  cache_positions=", t_cache_positions);
        DEBUG_LOG("VNode::execute  kvLen=", t_kvLen);
        /*
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
        auto mask_length = t_attention_mask.size(-1);
        // raw_causal_mask is already ensured to be triu by transformation
        // attention_mask            : i64[N, kv_len]
        //      0 means mask-out, 1 means attends to
        // dst [batch_size, 1, q_len, kvLen]
        auto batch_size = *t_batch_size.ptr<int32_t>(0);
        auto kvLen = *t_kvLen.ptr<int32_t>(0);
        auto qLen = t_cache_positions.size(0);
        auto* prow = t_cache_positions.ptr<int32_t>(0);
        T min_dtype = std::numeric_limits<T>::lowest();
        for (int32_t n = 0; n < batch_size; n++) {
            auto* pamask = t_attention_mask.ptr<int32_t>(n, 0);
            auto* pdst = t_dst.ptr<T>(n, 0);
            for (size_t i = 0; i < qLen; i++, pdst += kvLen) {
                auto row = prow[i];
                // < mask_length ?
                for (int32_t j = 0; j < kvLen; j++) {
                    bool cmask_eq0 = (j <= row);
                    bool amask_eq0 = (pamask[j] == 0);
                    bool padding_mask = (cmask_eq0 && amask_eq0);
                    pdst[j] = (padding_mask | (!cmask_eq0)) ? min_dtype : T(0);
                }
            }
        }
        DEBUG_LOG("VNode::execute  dst=", t_dst);
    }
};

void VNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto dstPrecision = getOriginalOutputPrecisionAtPort(0);

    if (m_config.type == "CausalMaskPreprocess") {
        /*
            inputs:
                attention_mask            : i64[N, kv_len]
                                            0 means mask-out, 1 means attends to
                batch_size (size_Gather)  : i32[1]
                cache_positions  i32[q_len];
                kvLen            i32[1];

            outputs:
                causal mask for SDPA : f32[batch_size, 1, q_len, kvLen]
        */
        if (dstPrecision == ov::element::bf16) {
            m_executor = std::make_shared<VNodeExecutorCausalMaskPreprocess<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<VNodeExecutorCausalMaskPreprocess<float>>();
            dstPrecision = ov::element::f32;
        }

        // initialize input ports
        std::vector<PortConfigurator> inPortConfigs;
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(0), false, -1);
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(1), false, -1);
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(2), false, -1);
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::i32, getInputShapeAtPort(3), false, -1);

        // initialize output port
        std::vector<PortConfigurator> outPortConfigs;
        outPortConfigs.emplace_back(LayoutType::ncsp, dstPrecision, getOutputShapeAtPort(0), false, -1);

        addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);

    } else {
        OPENVINO_THROW("CPU: VNode type not supported : " + m_config.type);
    }
}

void VNode::execute(dnnl::stream strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size()), outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getSrcMemoryAtPort(i);
    }

    if (m_config.type == "CausalMaskPreprocess") {
        auto batch_size = *(inputs[1]->getDataAs<uint32_t>());
        auto qLen = inputs[2]->getStaticDims()[0];
        auto kvLen = *(inputs[3]->getDataAs<uint32_t>());
        VectorDims newDims{batch_size, 1, qLen, kvLen};
        DEBUG_LOG("VNode::execute", m_config.type, "  batch_size=", batch_size, " qLen=", qLen, " kvLen=", kvLen);
        redefineOutputMemory({newDims});
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getDstMemoryAtPort(i);
    }
    m_executor->execute(strm, m_config, inputs, outputs);
}

bool VNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const intel_cpu::VNode>(op);
        if (!node) {
            errorMessage = "Only VNode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
