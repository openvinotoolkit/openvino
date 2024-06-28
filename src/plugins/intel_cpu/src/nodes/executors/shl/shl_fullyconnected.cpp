// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shl_fullyconnected.hpp"

#include "csinn/csi_nn.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

bool ShlFCExecutor::supports(const FCConfig& config) {
    if (config.attrs.weightsNonTransposed) {
        DEBUG_LOG("ShlFCExecutor: weightsNonTransposed is not supported!");
        return false;
    }

    if (!config.postOps.empty()) {
        DEBUG_LOG("ShlFCExecutor: PostOps are not supported");
        return false;
    }

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);
    if (!everyone_is(ov::element::f32, srcDesc->getPrecision(), weiDesc->getPrecision(), dstDesc->getPrecision())) {
        DEBUG_LOG("ShlFCExecutor: supports only f32");
        return false;
    }

    if (config.attrs.withBias) {
        const auto& biaDesc = config.descs.at(ARG_BIAS);
        if (biaDesc->getPrecision() != ov::element::f32) {
            DEBUG_LOG("ShlFCExecutor: supports only f32 bias");
            return false;
        }

        const auto& biasDims = biaDesc->getShape().getStaticDims();
        const auto& outDims = dstDesc->getShape().getDims();
        const bool isByChannel = biasDims.back() == outDims.back();
        if (!isByChannel || !std::all_of(biasDims.begin(), biasDims.end() - 1, [](const Dim dim) { return dim == 1; })) {
            DEBUG_LOG("ShlFCExecutor: only 'by channel' bias is supported");
            return false;
        }
    }

    return true;
}

ShlFCExecutor::ShlFCExecutor(const FCAttrs& attrs,
                             const PostOps& postOps,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr context) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    // Allocate Shl session
    sess = ShlSession(CSINN_RM_LAYER);

    // Allocate Shl tensors
    src = ShlTensor(sess, precisionToShlDataType(srcDesc->getPrecision()), getShlDataLayoutByMemoryDesc(srcDesc));
    wei = ShlTensor(sess, precisionToShlDataType(weiDesc->getPrecision()), getShlDataLayoutByMemoryDesc(weiDesc, true));
    dst = ShlTensor(sess, precisionToShlDataType(dstDesc->getPrecision()), getShlDataLayoutByMemoryDesc(dstDesc));
    bias = ShlTensor(sess);

    if (attrs.withBias) {
        const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
        bias = ShlTensor(sess, memory.at(ARG_BIAS)->getDescPtr()->getShape().getStaticDims(),
                        precisionToShlDataType(biasDesc->getPrecision()),
                        getShlDataLayoutByMemoryDesc(biasDesc), memory.at(ARG_BIAS)->getData());
    } else {
        bias = ShlTensor(sess);
    }

    // Init FC params
    params = ShlFCParams(sess, CSINN_RVV);

    OPENVINO_ASSERT(csinn_fullyconnected_init(src.get(), dst.get(), wei.get(), bias.get(), params.get()) == CSINN_TRUE,
                    "ShlFCExecutor: failed to init FC");
}

bool ShlFCExecutor::update(const MemoryArgs& memory) {
    src.setShape(memory.at(ARG_SRC)->getDescPtr()->getShape().getStaticDims());
    wei.setShape(memory.at(ARG_WEI)->getDescPtr()->getShape().getStaticDims());
    dst.setShape(memory.at(ARG_DST)->getDescPtr()->getShape().getStaticDims());
    return true;
}

void ShlFCExecutor::execute(const MemoryArgs& memory) {
    src.setData(memory.at(ARG_SRC)->getData());
    wei.setData(memory.at(ARG_WEI)->getData());
    dst.setData(memory.at(ARG_DST)->getData());

    OPENVINO_ASSERT(csinn_fullyconnected(src.get(), dst.get(), wei.get(), bias.get(), params.get()) == CSINN_TRUE,
                    "ShlFCExecutor: failed to execute");
}

}  // namespace intel_cpu
}  // namespace ov
