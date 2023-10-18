// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <chrono>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ie_ngraph_utils.hpp>
#include <string>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "emitters/x64/jit_dnnl_emitters.hpp"
#include "emitters/x64/jit_load_store_emitters.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "utils/plain_tensor.hpp"

using namespace InferenceEngine;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

#if defined(OPENVINO_ARCH_X86_64)

#endif  // OPENVINO_ARCH_X86_64

RoPE::RoPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto node = std::dynamic_pointer_cast<const RoPENode>(op);
    m_config = node->get_config();
}

template <typename T>
struct RoPEExecutor : public RoPE::Executor {
    void execute(RoPE* pnode) override {
        ov::intel_cpu::PlainTensor<T> t_src(pnode->getParentEdgeAt(0)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_cos(pnode->getParentEdgeAt(1)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_sin(pnode->getParentEdgeAt(2)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<int32_t> gather;
        ov::intel_cpu::PlainTensor<T> t_past;
        auto& config = pnode->getConfig();
        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(3, config.slice_start, config.slice_stop);
        }
        if (config.input_trans0213) {
            t_src = t_src.permute({0, 2, 1, 3});
        }
        if (config.gather_position_arg_id > 0) {
            gather.reset(pnode->getParentEdgeAt(config.gather_position_arg_id)->getMemoryPtr());
        }

        auto batch_size = t_src.size(0);
        auto head_cnt = t_src.size(1);
        auto seq_len = t_src.size(2);
        auto feature_size = t_src.size(3);

        size_t past_len = 0;
        if (config.concat_with_past_arg_id > 0) {
            t_past.reset(pnode->getParentEdgeAt(config.concat_with_past_arg_id)->getMemoryPtr());
            past_len = t_past.size(2);
        }

        VectorDims result_shape{batch_size, head_cnt, past_len + seq_len, feature_size};

        pnode->redefineOutputMemory({result_shape});

        ov::intel_cpu::PlainTensor<T> t_dst(pnode->getChildEdgeAt(0)->getMemoryPtr());

        auto rotary_dims = config.ndims;
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
            auto* dst = &t_dst.at({b, h, past_len + p, 0});

            if (past_len) {
                memcpy(&t_dst.at({b, h, 0, 0}), &t_past.at({b, h, 0, 0}), past_len * feature_size * sizeof(T));
            }

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
struct RoPEExecutor2 : public RoPE::Executor {
    void execute(RoPE* pnode) override {
        ov::intel_cpu::PlainTensor<T> t_src(pnode->getParentEdgeAt(0)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_cos(pnode->getParentEdgeAt(1)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_sin(pnode->getParentEdgeAt(2)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<T> t_past;

        auto& config = pnode->getConfig();

        // B,L,H,S
        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = t_src.size(2);
        auto feature_size = t_src.size(3);

        size_t past_len = 0;
        if (config.concat_with_past_arg_id > 0) {
            t_past.reset(pnode->getParentEdgeAt(config.concat_with_past_arg_id)->getMemoryPtr());
            past_len = t_past.size(2);
        }

        VectorDims result_shape{batch_size, head_cnt, past_len + seq_len, feature_size};

        pnode->redefineOutputMemory({result_shape});

        ov::intel_cpu::PlainTensor<T> t_dst(pnode->getChildEdgeAt(0)->getMemoryPtr());

        auto rotary_dims = config.ndims;

        int mode = 0;
        mode |= config.sin_is_raw3d ? 0x01 : 0x0;
        mode |= config.cos_is_raw3d ? 0x10 : 0x0;

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            auto* x = &t_src.at({b, p, h, 0});
            float* cos = config.cos_is_raw3d ? &t_cos.at({b, p, 0}, true) : &t_cos.at({b, p, h, 0}, true);
            float* sin = config.sin_is_raw3d ? &t_sin.at({b, p, 0}, true) : &t_sin.at({b, p, h, 0}, true);
            auto* dst = &t_dst.at({b, h, past_len + p, 0});

            if (past_len) {
                memcpy(&t_dst.at({b, h, 0, 0}), &t_past.at({b, h, 0, 0}), past_len * feature_size * sizeof(T));
            }

            size_t i = 0;
            size_t j = 0;
            // sin* (-x[1], x[0], -x[3], x[2], -x[5], x[4], ..... )
            switch (mode) {
            case 0x00:
                for (; i < rotary_dims; i += 2) {
                    dst[i] = cos[i] * x[i] + sin[i] * (-x[i + 1]);
                    dst[i + 1] = cos[i + 1] * x[i + 1] + sin[i + 1] * (x[i]);
                }
                break;
            case 0x01:
                for (; i < rotary_dims; i += 2, j++) {
                    dst[i] = cos[i] * x[i] + sin[j] * (-x[i + 1]);
                    dst[i + 1] = cos[i + 1] * x[i + 1] + sin[j] * (x[i]);
                }
                break;
            case 0x10:
                for (; i < rotary_dims; i += 2, j++) {
                    dst[i] = cos[j] * x[i] + sin[i] * (-x[i + 1]);
                    dst[i + 1] = cos[j] * x[i + 1] + sin[i + 1] * (x[i]);
                }
                break;
            case 0x11:
                for (; i < rotary_dims; i += 2, j++) {
                    dst[i] = cos[j] * x[i] + sin[j] * (-x[i + 1]);
                    dst[i + 1] = cos[j] * x[i + 1] + sin[j] * (x[i]);
                }
                break;
            }
            for (; i < feature_size; i++) {
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
    auto CosSinPrecision = InferenceEngine::Precision::FP32;  // rtPrecision

    if (m_config.output_trans0213) {
        assert(m_config.input_trans0213 == false);
        assert(m_config.slice_start == 0);
        assert(m_config.slice_stop == 0);
        assert(m_config.gather_position_arg_id == 0);
        if (rtPrecision == InferenceEngine::Precision::BF16) {
            m_executor = std::make_shared<RoPEExecutor2<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutor2<float>>();
        }
    } else {
        assert(m_config.sin_is_raw3d == false);
        assert(m_config.cos_is_raw3d == false);
        if (rtPrecision == InferenceEngine::Precision::BF16) {
            m_executor = std::make_shared<RoPEExecutor<ov::bfloat16>>();
        } else {
            m_executor = std::make_shared<RoPEExecutor<float>>();
        }
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(1), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(2), false, -1);
    if (m_config.gather_position_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   InferenceEngine::Precision::I32,
                                   getInputShapeAtPort(m_config.gather_position_arg_id),
                                   false,
                                   -1);
    }
    if (m_config.concat_with_past_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   rtPrecision,
                                   getInputShapeAtPort(m_config.concat_with_past_arg_id),
                                   false,
                                   -1);
    }
    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void RoPE::execute(dnnl::stream strm) {
    m_executor->execute(this);
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
