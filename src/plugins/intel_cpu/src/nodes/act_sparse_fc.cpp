// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "act_sparse_fc.h"

#include "common/arbitrary_order_desc_creator.h"
#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "nodes/reorder.h"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_X86_64)
#    include "kernels/x64/act_sparse_fc_kernel.hpp"
#endif

#include "openvino/core/parallel.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace ov {
namespace intel_cpu {
namespace node {

void ActSparseFC::execute(dnnl::stream strm) {
    MAYBE_UNUSED(strm);
    if (m_executor) {
        const auto* input = getSrcDataAtPortAs<float>(0);
        const auto* weight = m_weight->getDataAs<uint8_t>();
        const auto* zp = m_config.with_zero_point ? m_zp->getDataAs<uint8_t>() : nullptr;
        const auto* scales = m_config.is_quantized ? m_scales->getDataAs<float>() : nullptr;
        auto* output = getDstDataAtPortAs<float>(0);

        const auto& ishape = getSrcMemoryAtPort(0)->getStaticDims();
        int M = shape_size(ishape) / ishape[ishape.size() - 1];

        (*m_executor)(input, output, M, m_config.ic, m_config.oc, m_config.threshold, 0, weight, scales, zp);
    }
}

ActSparseFC::ActSparseFC(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;

    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    const auto node = std::dynamic_pointer_cast<const ActSparseFCNode>(op);
    m_config = node->get_config();
}

struct ActSparseFCKey {
    bool is_quantized;
    bool is_int4;
    bool with_zero_point;
    int ic_q_group_size;

    size_t hash() const {
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        seed = hash_combine(seed, is_quantized);
        seed = hash_combine(seed, is_int4);
        seed = hash_combine(seed, with_zero_point);
        seed = hash_combine(seed, ic_q_group_size);
        return seed;
    }

    bool operator==(const ActSparseFCKey& rhs) const {
        return is_quantized == rhs.is_quantized && is_int4 == rhs.is_int4 && with_zero_point == rhs.with_zero_point &&
               ic_q_group_size == rhs.ic_q_group_size;
    }
};

void ActSparseFC::createPrimitive() {
    ActSparseFCKey key;
    key.is_quantized = m_config.is_quantized;
    key.is_int4 = m_config.is_int4;
    key.with_zero_point = m_config.with_zero_point;
    key.ic_q_group_size = m_config.ic_q_group_size;

    auto buildExecutor = [this](const ActSparseFCKey& key) -> std::shared_ptr<ActSparseFcKernel> {
#if defined(OPENVINO_ARCH_X86_64)
        return std::make_shared<ActSparseFcKernel>(context->getScratchPad(),
                                                   key.is_quantized,
                                                   key.is_int4,
                                                   key.with_zero_point,
                                                   key.ic_q_group_size);
#else
        return nullptr;
#endif
    };

    m_executor = nullptr;
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, buildExecutor);
    m_executor = result.first;

    if (!m_executor)
        OPENVINO_THROW("Failed to create executor for node ", getName(), ".");

    // reorder weights
    const auto& engine = getEngine();

    auto create_weight = [&]() {
        auto raw_weight_mem = getSrcMemoryAtPort(1);
        MemoryPtr weight_mem;
        if (m_config.is_int4) {
            // weight : [OC, IC/group_size, group_size] => [IC, OC/2, 2]
            // each row is further reordered in unit of 16 x i4 in [0,8,1,9,2,a,3,b,4,c,5,d,6,e,7,f] order
            weight_mem = std::make_shared<Memory>(engine, raw_weight_mem->getDescPtr());

            const auto& dims = raw_weight_mem->getShape().getStaticDims();
            OPENVINO_ASSERT(dims.size() == 3);
            OPENVINO_ASSERT(dims[0] == static_cast<size_t>(m_config.oc));
            OPENVINO_ASSERT(dims[1] == static_cast<size_t>(m_config.ic / m_config.ic_q_group_size));
            OPENVINO_ASSERT(dims[2] == static_cast<size_t>(m_config.ic_q_group_size));

            auto* src = raw_weight_mem->getDataAs<uint8_t>();
            auto* dst = weight_mem->getDataAs<uint8_t>();
            m_executor->repack_weights_i4(src, dst, m_config.ic, m_config.oc);
        } else {
            // raw [OC, IC] layout
            // target [IC, OC] layout
            ArbitraryOrderDescCreator descCreator({1, 0});
            auto dst_mem_desc =
                descCreator.createSharedDesc(raw_weight_mem->getPrecision(), raw_weight_mem->getShape());

            weight_mem = std::make_shared<Memory>(engine, dst_mem_desc);
            node::Reorder::reorderData(*raw_weight_mem, *weight_mem, context->getParamsCache());
        }
        return weight_mem;
    };

    auto create_zp_i4 = [&]() {
        // [OC, IC/group_size, 1] => [IC/group_size, OC]
        auto raw_zp_mem = getSrcMemoryAtPort(3);
        auto zp_mem = std::make_shared<Memory>(engine, raw_zp_mem->getDescPtr());

        auto* src = raw_zp_mem->getDataAs<uint8_t>();
        auto* dst = zp_mem->getDataAs<uint8_t>();

        m_executor->repack_weights_i4(src, dst, m_config.ic / m_config.ic_q_group_size, m_config.oc);
        return zp_mem;
    };

    auto create_scales_i4 = [&]() {
        // [OC, IC/group_size, 1] => [IC/group_size, OC]
        auto raw_scales_mem = getSrcMemoryAtPort(2);
        ArbitraryOrderDescCreator descCreator({2, 1, 0});
        auto dst_mem_desc = descCreator.createSharedDesc(raw_scales_mem->getPrecision(), raw_scales_mem->getShape());

        auto scales_mem = std::make_shared<Memory>(engine, dst_mem_desc);
        node::Reorder::reorderData(*raw_scales_mem, *scales_mem, context->getParamsCache());
        return scales_mem;
    };

    if (!m_config.is_int4) {
        // int8 is perOC, no need for reorder
        if (m_config.is_quantized)
            m_scales = getSrcMemoryAtPort(2);
        if (m_config.with_zero_point)
            m_zp = getSrcMemoryAtPort(3);
    }

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const auto string_hash = getOriginalLayers() + std::to_string(m_config.is_int4);
        m_weight = *weightCache->findOrCreate(string_hash + "_weight", create_weight);
        if (m_config.is_int4) {
            if (m_config.with_zero_point)
                m_zp = *weightCache->findOrCreate(string_hash + "_zp_i4", create_zp_i4);
            if (m_config.is_quantized)
                m_scales = *weightCache->findOrCreate(string_hash + "_scales_i4", create_scales_i4);
        }
    } else {
        m_weight = create_weight();
        if (m_config.is_int4) {
            if (m_config.with_zero_point)
                m_zp = create_zp_i4();
            if (m_config.is_quantized)
                m_scales = create_scales_i4();
        }
    }
}

void ActSparseFC::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    // OPENVINO_ASSERT(rtPrecision == ov::element::f32, "Unexpected rtPrecision:", rtPrecision);
    auto rtPrecision = ov::element::f32;

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);  // input
    inPortConfigs.emplace_back(LayoutType::ncsp,
                               getOriginalInputPrecisionAtPort(1),
                               getInputShapeAtPort(1),
                               false,
                               -1);  // weight
    if (m_config.is_quantized) {
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f32, getInputShapeAtPort(2), false, -1);  // scales
        if (m_config.with_zero_point)
            inPortConfigs.emplace_back(LayoutType::ncsp,
                                       getOriginalInputPrecisionAtPort(3),
                                       getInputShapeAtPort(3),
                                       false,
                                       -1);  // zero-pt
    }

    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

bool ActSparseFC::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        const auto node = std::dynamic_pointer_cast<const ActSparseFCNode>(op);
        const auto& config = node->get_config();
        if ((config.oc % 32) > 0) {
            errorMessage = "Unsupported OC size for node " + node->get_friendly_name();
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
