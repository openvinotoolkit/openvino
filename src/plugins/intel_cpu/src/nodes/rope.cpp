// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope.h"

#include <chrono>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <ie_ngraph_utils.hpp>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "utils/plain_tensor.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

RoPE::RoPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const RoPENode>(op);
    m_config = node->get_config();
}

template <typename T>
struct RoPE::RoPEExecutorRotateHalf : public RoPE::Executor {
    void execute(dnnl::stream strm,
                 const RoPENode::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor<T> t_src(inputs[0]);
        ov::intel_cpu::PlainTensor<float> t_cos(inputs[1]);
        ov::intel_cpu::PlainTensor<float> t_sin(inputs[2]);
        ov::intel_cpu::PlainTensor<T> t_dst(outputs[0]);
        ov::intel_cpu::PlainTensor<int32_t> gather;

        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(3, config.slice_start, config.slice_stop);
        }
        if (config.input_trans0213) {
            t_src = t_src.permute({0, 2, 1, 3});
        }
        if (config.gather_position_arg_id > 0) {
            gather.reset(inputs[config.gather_position_arg_id]);
        }

        auto batch_size = t_src.size(0);
        auto head_cnt = t_src.size(1);
        auto seq_len = t_src.size(2);
        auto feature_size = t_src.size(3);

        auto rotary_dims = config.rotary_ndims;
        auto half_rotary_dims = rotary_dims / 2;

        parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p;
            if (gather) {
                if (gather.m_rank == 4)
                    cos_pos = gather.at({b, h, p, 0}, true);
                else
                    cos_pos = gather.at({b, p}, true);
            }
            auto* src = &t_src.at({b, h, p, 0});
            auto* cos = &t_cos.at({b, h, cos_pos, 0}, true);
            auto* sin = &t_sin.at({b, h, cos_pos, 0}, true);
            auto* dst = &t_dst.at({b, h, p, 0});

            size_t i = 0;
            for (; i < half_rotary_dims; i++) {
                dst[i] = cos[i] * src[i] + sin[i] * (-src[i + half_rotary_dims]);
            }
            for (; i < rotary_dims; i++) {
                dst[i] = cos[i] * src[i] + sin[i] * (src[i - half_rotary_dims]);
            }
            for (; i < feature_size; i++) {
                dst[i] = src[i];
            }
        });
    }
};

template <typename T>
struct RoPE::RoPEExecutorInterleaved : public RoPE::Executor {
    void execute(dnnl::stream strm,
                 const RoPENode::Config& config,
                 const std::vector<MemoryPtr>& inputs,
                 const std::vector<MemoryPtr>& outputs) override {
        ov::intel_cpu::PlainTensor<T> t_src(inputs[0]);
        ov::intel_cpu::PlainTensor<float> t_sin_cos(inputs[1]);
        ov::intel_cpu::PlainTensor<T> t_dst(outputs[0]);

        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = t_src.size(2);
        auto head_dims = t_src.size(3);

        auto rotary_dims = config.rotary_ndims;
        auto half_rotary_dims = rotary_dims / 2;
        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            auto* x = &t_src.at({b, p, h, 0});
            float* sin = &t_sin_cos.at({b, p, 0}, true);
            float* cos = &t_sin_cos.at({b, p, half_rotary_dims}, true);
            auto* dst = &t_dst.at({b, h, p, 0});

            size_t i = 0;
            for (size_t j = 0; i < rotary_dims; i += 2, j++) {
                dst[i] = cos[j] * x[i] - sin[j] * x[i + 1];
                dst[i + 1] = cos[j] * x[i + 1] + sin[j] * x[i];
            }
            for (; i < head_dims; i++) {
                dst[i] = x[i];
            }
        });
    }
};

void RoPE::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);

    auto rtPrecision = srcPrecision;
    auto CosSinPrecision = ov::element::f32;

    if (m_config.is_interleaved) {
        OPENVINO_ASSERT(m_config.input_trans0213 == false);
        OPENVINO_ASSERT(m_config.slice_start == 0);
        OPENVINO_ASSERT(m_config.slice_stop == 0);
        OPENVINO_ASSERT(m_config.gather_position_arg_id == 0);
        if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorInterleaved<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorInterleaved<float>>();
            rtPrecision = ov::element::f32;
        }
    } else {
        if (rtPrecision == ov::element::bf16) {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutorRotateHalf<float>>();
            rtPrecision = ov::element::f32;
        }
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(1), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(2), false, -1);
    if (m_config.gather_position_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   ov::element::i32,
                                   getInputShapeAtPort(m_config.gather_position_arg_id),
                                   false,
                                   -1);
    }

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void RoPE::execute(dnnl::stream strm) {
    std::vector<MemoryPtr> inputs(getParentEdges().size()), outputs(getChildEdges().size());
    for (size_t i = 0; i < inputs.size(); i++) {
        inputs[i] = getParentEdgeAt(i)->getMemoryPtr();
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = getChildEdgeAt(i)->getMemoryPtr();
    }
    m_executor->execute(strm, m_config, inputs, outputs);
}

bool RoPE::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const RoPENode>(op);
        if (!node) {
            errorMessage = "Only RoPENode operation is supported";
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
