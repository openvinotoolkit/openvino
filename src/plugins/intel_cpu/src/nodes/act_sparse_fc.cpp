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

    Executor(ActSparseFC* pnode, DnnlScratchPadPtr scrachPad) : m_node(pnode), m_scrachPad(scrachPad) {}

    void execute() override {
        const auto* input = m_node->getSrcDataAtPortAs<float>(0);
        const auto* weight = m_node->getSrcDataAtPortAs<uint8_t>(1);
        const auto* zp = m_node->getSrcDataAtPortAs<uint8_t>(2);
        const auto* scales = m_node->getSrcDataAtPortAs<float>(3);
        auto* output = m_node->getDstDataAtPortAs<float>(0);

        const auto& ishape = m_node->getSrcMemoryAtPort(0)->getStaticDims();
        int M = shape_size(ishape) / ishape[ishape.size() - 1];

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

    NodeConfig config;

    for (int i = 0; i < 4; i++) {
        PortConfig dataConfig;
        ov::element::Type prec = rtPrecision;
        if (i == 1) prec = ov::element::u8; // fc_weight_u8
        if (i == 2) prec = ov::element::u8; // fc_weight_zero_point_u8
        if (i == 3) prec = ov::element::f32; // fc_weight_scales_per_OC
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        if (i == 1) {
            ArbitraryOrderDescCreator descCreator({1, 0});
            dataConfig.setMemDesc(descCreator.createSharedDesc(prec, getInputShapeAtPort(i)));
        } else {
            auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
            dataConfig.setMemDesc(descCreator->createSharedDesc(prec, getInputShapeAtPort(i)));
        }
        config.inConfs.push_back(dataConfig);
    }
    {
        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        auto descCreator = BlockedDescCreator::getCommonCreators().at(LayoutType::ncsp);
        dataConfig.setMemDesc(descCreator->createSharedDesc(rtPrecision, getOutputShapeAtPort(0)));
        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref_any});

    DEBUG_LOG(">>>>>>>>>", supportedPrimitiveDescriptors[0]);
    return;
#if 0
    DEBUG_LOG(">>>>>>>>>");
    if (!supportedPrimitiveDescriptors.empty())
        return;

    DEBUG_LOG(">>>>>>>>>");
    auto rtPrecision = getOriginalInputPrecisionAtPort(0);
    OPENVINO_ASSERT(rtPrecision == ov::element::f32, "Unexpected rtPrecision:", rtPrecision);


    DEBUG_LOG(">>>>>>>>>");
    std::vector<PortConfigurator> inPortConfigs;
    std::vector<PortConfigurator> outPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::nspc, ov::element::u8, getInputShapeAtPort(1), false, -1);  // fc_weight_u8
    inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::u8, getInputShapeAtPort(2), false, -1);  // fc_weight_zero_point_u8
    inPortConfigs.emplace_back(LayoutType::ncsp, ov::element::f32, getInputShapeAtPort(3), false, -1); // fc_weight_scales_per_OC

    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);    // output

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);

    DEBUG_LOG(">>>>>>>>>", supportedPrimitiveDescriptors[0]);
#endif
}

bool ActSparseFC::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
#if defined(OPENVINO_ARCH_X86_64)
    try {
        const auto node = std::dynamic_pointer_cast<const ActSparseFCNode>(op);
        const auto& config = node->get_config();
        if (config.ic_q_group_size > 0) {
            errorMessage = "Unsupported IC group size";
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
