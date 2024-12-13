// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "act_sparse_fc.h"

#include <string>
#include <vector>

#include "common/arbitrary_order_desc_creator.h"
#include "common/primitive_hashing_utils.hpp"
#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"
#include "nodes/reorder.h"

#if defined(OPENVINO_ARCH_X86_64)
#include "kernels/x64/act_sparse_fc_kernel.hpp"
#endif

#include "openvino/core/parallel.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_X86_64)

struct ActSparseFC::Executor : public ActSparseFC::ExecutorBase {
    ActSparseFC * m_node;
    DnnlScratchPadPtr m_scrachPad;
    MemoryPtr m_weight;
    MemoryPtr m_zp;
    MemoryPtr m_scales;
    ActSparseFCNode::Config& m_config;

    void show(const char * name, uint8_t * src, int stride, int rows, int cols) {
        printf("===== %s \n", name);
        for (int r = 0; r < rows; r++, src += stride) {
            for (int c = 0; c < cols; c++) {
                printf("%02X,", src[c]);
            }
            printf("\n");
        }
    }

    Executor(ActSparseFC* pnode, DnnlScratchPadPtr scrachPad) : m_node(pnode), m_scrachPad(scrachPad), m_config(m_node->m_config) {
        // reorder weights
        const auto& context = m_node->context;
        const auto& engine = m_node->getEngine();

        std::cout << m_node->getName() << std::endl;
        auto create_weight = [&]() {
            auto raw_weight_mem = m_node->getSrcMemoryAtPort(1);
            MemoryPtr weight_mem;
            if (m_config.is_int4) {
                // weight : [OC, IC/group_size, group_size] => [IC, OC/2, 2]
                // each row is further reordered in unit of 16 x i4 in [0,8,1,9,2,a,3,b,4,c,5,d,6,e,7,f] order
                weight_mem = std::make_shared<Memory>(engine, raw_weight_mem->getDescPtr());

                const auto& dims = raw_weight_mem->getShape().getStaticDims();
                OPENVINO_ASSERT(dims.size() == 3);
                OPENVINO_ASSERT(dims[0] == m_config.oc);
                OPENVINO_ASSERT(dims[1] == m_config.ic / m_config.ic_q_group_size);
                OPENVINO_ASSERT(dims[2] == m_config.ic_q_group_size);

                auto* src = raw_weight_mem->getDataAs<uint8_t>();
                auto* dst = weight_mem->getDataAs<uint8_t>();
                ov::Extensions::Cpu::XARCH::dynPruneLinear_repack_i4(src, dst, m_config.ic, m_config.oc);
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
            auto raw_zp_mem = m_node->getSrcMemoryAtPort(3);
            auto zp_mem = std::make_shared<Memory>(engine, raw_zp_mem->getDescPtr());

            auto* src = raw_zp_mem->getDataAs<uint8_t>();
            auto* dst = zp_mem->getDataAs<uint8_t>();

            ov::Extensions::Cpu::XARCH::dynPruneLinear_repack_i4(src, dst, m_config.ic/m_config.ic_q_group_size, m_config.oc);
            return zp_mem;
        };

        auto create_scales_i4 = [&]() {
            // [OC, IC/group_size, 1] => [IC/group_size, OC]
            auto raw_scales_mem = m_node->getSrcMemoryAtPort(2);
            ArbitraryOrderDescCreator descCreator({2, 1, 0});
            auto dst_mem_desc =
                descCreator.createSharedDesc(raw_scales_mem->getPrecision(), raw_scales_mem->getShape());

            auto scales_mem = std::make_shared<Memory>(engine, dst_mem_desc);
            node::Reorder::reorderData(*raw_scales_mem, *scales_mem, context->getParamsCache());
            return scales_mem;
        };

        if (!m_config.is_int4) {
            // int8 is perOC, no need for reorder
            if (m_config.is_quantized)
                m_scales = m_node->getSrcMemoryAtPort(2);
            if (m_config.with_zero_point)
                m_zp = m_node->getSrcMemoryAtPort(3);
        }

        auto weightCache = context->getWeightsCache();
        if (weightCache != nullptr) {
            const auto string_hash = m_node->getOriginalLayers() + std::to_string(m_config.is_int4);
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

    void execute() override {
        const auto* input = m_node->getSrcDataAtPortAs<float>(0);
        const auto* weight = m_weight->getDataAs<uint8_t>();
        const auto* zp = m_config.with_zero_point ?  m_zp->getDataAs<uint8_t>() : nullptr;
        const auto* scales = m_config.is_quantized ? m_scales->getDataAs<float>() : nullptr;
        auto* output = m_node->getDstDataAtPortAs<float>(0);

        const auto& ishape = m_node->getSrcMemoryAtPort(0)->getStaticDims();
        int M = shape_size(ishape) / ishape[ishape.size() - 1];

        if (m_config.is_quantized) {
            if (m_config.is_int4) {
                ov::Extensions::Cpu::XARCH::dynPruneLinear_i4(input,
                                                            m_config.threshold,
                                                            0,
                                                            weight,
                                                            zp,
                                                            scales,
                                                            output,
                                                            M,
                                                            m_config.ic,
                                                            m_config.oc,
                                                            m_config.ic_q_group_size);
            } else {
                ov::Extensions::Cpu::XARCH::dynPruneLinear_i8(input,
                                                            m_node->m_config.threshold,
                                                            0,
                                                            weight,
                                                            zp,
                                                            scales,
                                                            output,
                                                            M,
                                                            m_node->m_config.ic,
                                                            m_node->m_config.oc);
            }
        } else {
            ov::Extensions::Cpu::XARCH::dynPruneLinear_f16(input,
                                                           m_config.threshold,
                                                           0,
                                                           reinterpret_cast<const ov::float16*>(weight),
                                                           output,
                                                           M,
                                                           m_config.ic,
                                                           m_config.oc);
        }
    }
};
#else
struct ActSparseFC::Executor : public ActSparseFC::ExecutorBase {
    ActSparseFC * m_pnode;
    Executor(ActSparseFC * pnode, DnnlScratchPadPtr scrachPad) : m_pnode(pnode) {}
    void execute() override {}
};
#endif

void ActSparseFC::createPrimitive() {
    auto rtPrecision = getInputPrecisions()[0];
#ifdef OPENVINO_ARCH_X86_64
    m_executor = std::make_shared<Executor>(this, context->getScratchPad());
#endif

    if (!m_executor) {
        OPENVINO_THROW("ActSparseFC Executor creation fails with precision " + rtPrecision.to_string());
    }
}

void ActSparseFC::execute(dnnl::stream strm) {
    MAYBE_UNUSED(strm);
    m_executor->execute();
}

ActSparseFC::ActSparseFC(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;

    const auto & config = context->getConfig();

    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }
    const auto node = std::dynamic_pointer_cast<const ActSparseFCNode>(op);
    m_config = node->get_config();
}

void ActSparseFC::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    OPENVINO_ASSERT(rtPrecision == ov::element::f32, "Unexpected rtPrecision:", rtPrecision);

    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;

    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::ncsp, getOriginalInputPrecisionAtPort(1), getInputShapeAtPort(1), false, -1);  // weight
    if (m_config.is_quantized) {
        inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f32, getInputShapeAtPort(2), false, -1);  // scales
        if (m_config.with_zero_point)
            inPortConfigs.emplace_back(LayoutType::ncsp, getOriginalInputPrecisionAtPort(3), getInputShapeAtPort(3), false, -1);  // zero-pt
    }

    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

bool ActSparseFC::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        const auto node = std::dynamic_pointer_cast<const ActSparseFCNode>(op);
        const auto& config = node->get_config();
        if ((config.oc % 16) > 0) {
            errorMessage = "Unsupported OC size for node " + node->get_friendly_name();
            std::cout << "ActSparseFC: " << errorMessage << std::endl;
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
