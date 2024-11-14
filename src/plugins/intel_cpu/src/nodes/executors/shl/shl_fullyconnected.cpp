// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shl_fullyconnected.hpp"

#include "csinn/csi_nn.h"
#include "rvv/rvv.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {
namespace {
static MemoryPtr prepareWeightMemory(const MemoryPtr weightsMemory, const ExecutorContext::CPtr context) {
    DEBUG_LOG("ShlFCExecutor: prepack weights");

    auto create = [&]() {
        const auto& weiDesc = weightsMemory->getDescPtr();
        MemoryPtr _ptr = std::make_shared<Memory>(context->getEngine(),
                                                  intel_cpu::CpuBlockedMemoryDesc(ov::element::f32, weightsMemory->getShape()));
        cpu_parallel_memcpy(_ptr->getData(), weightsMemory->getData(), weightsMemory->getSize());
        DEBUG_LOG("ShlFCExecutor: cache miss, perform packing");
        const auto repack_wei = ShlTensor(ShlSession(), precisionToShlDataType(weiDesc->getPrecision()), getShlDataLayoutByMemoryDesc(weiDesc, true),
                                          weiDesc->getShape().getStaticDims(), _ptr->getData());
        shl_rvv_fc_gemm_reorder_weight_fp32(repack_wei.get());
        return _ptr;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const auto& wgtDims = weightsMemory->getStaticDims();
        std::string format = "gemm_shl_" + std::to_string(wgtDims[0]) + "_" + std::to_string(wgtDims[1]);
        const std::string string_hash = format + "_" + std::to_string(weightsMemory->getSize()) + "_" +
                                        std::to_string(reinterpret_cast<uint64_t>(weightsMemory->getData()));
        DEBUG_LOG("ShlFCExecutor: findOrCreate, string_hash: ", string_hash);
        return *weightCache->findOrCreate(string_hash, create);
    }

    DEBUG_LOG("ShlFCExecutor: Weights cache is not available");
    return create();
}
} // namespace

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
                             const ExecutorContext::CPtr context)
    : packedWeights(prepareWeightMemory(memory.at(ARG_WEI), context)) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    // Allocate Shl session
    sess = ShlSession();

    // Allocate Shl tensors
    src = ShlTensor(sess, precisionToShlDataType(srcDesc->getPrecision()), getShlDataLayoutByMemoryDesc(srcDesc));
    wei = ShlTensor(sess, precisionToShlDataType(weiDesc->getPrecision()), getShlDataLayoutByMemoryDesc(weiDesc, true),
                          weiDesc->getShape().getStaticDims());
    dst = ShlTensor(sess, precisionToShlDataType(dstDesc->getPrecision()), getShlDataLayoutByMemoryDesc(dstDesc));

    if (attrs.withBias) {
        const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
        bias = ShlTensor(sess, precisionToShlDataType(biasDesc->getPrecision()), getShlDataLayoutByMemoryDesc(biasDesc),
                               biasDesc->getShape().getStaticDims());
        with_bias = true;
    } else {
        bias = ShlTensor(sess);
    }

    // Init FC params
    params = ShlFCParams(sess, CSINN_RVV);

    OPENVINO_ASSERT(csinn_fullyconnected_init(src.get(), dst.get(), wei.get(), bias.get(), static_cast<csinn_fc_params*>(params.get())) == CSINN_TRUE,
                    "ShlFCExecutor: failed to init FC");
}

bool ShlFCExecutor::update(const MemoryArgs& memory) {
    // Weights and Bias have static shapes - no need to update them here
    src = src.cloneWithNewShape(memory.at(ARG_SRC)->getDescPtr()->getShape().getStaticDims());
    dst = dst.cloneWithNewShape(memory.at(ARG_DST)->getDescPtr()->getShape().getStaticDims());

    const auto src_shape = src.getShape();
    const auto dst_shape = dst.getShape();
    dim_M = std::accumulate(dst_shape.rbegin() + 1, dst_shape.rend(), size_t(1), std::multiplies<size_t>());
    dim_In = src_shape.back();
    dim_Out = dst_shape.back();
    LDA = dim_In * memory.at(ARG_SRC)->getPrecision().size();
    LDC = dim_Out * memory.at(ARG_DST)->getPrecision().size();

    return true;
}

void ShlFCExecutor::execute(const MemoryArgs& memory) {
    wei.setData(packedWeights->getData());
    if (with_bias) {
        bias.setData(memory.at(ARG_BIAS)->getData());
    }

    const auto nthreads = std::min(static_cast<int>(dim_M), parallel_get_max_threads());
    parallel_nt(nthreads, [&](const int ithr, const int nthr) {
        size_t dim_M0 = 0, dim_M1 = 0;
        splitter(dim_M, nthr, ithr, dim_M0, dim_M1);

        const auto M = dim_M1 - dim_M0;
        auto src_tensor = src.cloneWithNewShape(ov::Shape{ M, dim_In });
        auto dst_tensor = dst.cloneWithNewShape(ov::Shape{ M, dim_Out });
        src_tensor.setData(reinterpret_cast<uint8_t*>(memory.at(ARG_SRC)->getData()) + dim_M0 * LDA);
        dst_tensor.setData(reinterpret_cast<uint8_t*>(memory.at(ARG_DST)->getData()) + dim_M0 * LDC);

        OPENVINO_ASSERT(csinn_fullyconnected(src_tensor.get(), dst_tensor.get(), wei.get(), bias.get(), static_cast<csinn_fc_params*>(params.get())) == CSINN_TRUE,
                        "ShlFCExecutor: failed to execute");
    });
}

}  // namespace intel_cpu
}  // namespace ov
