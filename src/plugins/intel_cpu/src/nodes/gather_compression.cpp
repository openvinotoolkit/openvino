// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_compression.h"

#include "common/cpu_memcpy.h"
#include "ov_ops/gather_compressed.hpp"
#include "utils/ngraph_utils.hpp"

using namespace dnnl::impl::cpu;

#define THROW_ERROR(...) OPENVINO_THROW(getTypeStr(), " node with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherCompression::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gather_compression = std::dynamic_pointer_cast<const ov::op::internal::GatherCompressed>(op);
        if (!gather_compression) {
            errorMessage = "Only GatherCompression operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GatherCompression::GatherCompression(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (op->get_input_size() != 4 || op->get_output_size() != 1)
        THROW_ERROR("has incorrect number of input/output edges!");
}

void GatherCompression::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(GATHER_DATA);
    if (!(dataPrecision == ov::element::u4 || dataPrecision == ov::element::u8)) {
        THROW_ERROR(" has unsupported 'data' input precision: ", dataPrecision);
    }
    ov::element::Type zpPrecision = getOriginalInputPrecisionAtPort(GATHER_ZP);
    if (zpPrecision != ov::element::f32) {
        THROW_ERROR(" has unsupported 'zp' input precision: ", zpPrecision);
    }
    ov::element::Type scalePrecision = getOriginalInputPrecisionAtPort(GATHER_SCALE);
    if (scalePrecision != ov::element::f32) {
        THROW_ERROR(" has unsupported 'scale' input precision: ", scalePrecision);
    }

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, zpPrecision},
                          {LayoutType::ncsp, scalePrecision},
                          {LayoutType::ncsp, ov::element::i32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         ref_any);
}

bool GatherCompression::needPrepareParams() const {
    return false;
}

void GatherCompression::execute(dnnl::stream strm) {
    execReference();
}

void GatherCompression::executeDynamicImpl(dnnl::stream strm) {
    execReference();
}

void GatherCompression::execReferenceU4() {
    DEBUG_LOG(getName(), "execReferenceU4");
    auto srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    auto idxMemPtr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    const auto* psrc = srcMemPtr->getDataAs<uint8_t>();
    const auto* pidx = idxMemPtr->getDataAs<int32_t>();

    bool one_dim_zp = getParentEdgeAt(GATHER_ZP)->getMemoryPtr()->getShape().getRank() == 1;
    const auto* zp = getSrcDataAtPortAs<float_t>(GATHER_ZP);
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    auto* pdst = getDstDataAtPortAs<float>(0);

    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto batch = idxDims[0];
    const auto seqLen = idxDims[1];

    auto axisDim = srcMemPtr->getStaticDims()[0];
    auto groupDim = srcMemPtr->getStaticDims().size() == 2 ? 1 : srcMemPtr->getStaticDims()[1];
    auto feaDim = srcMemPtr->getStaticDims().size() == 2 ? srcMemPtr->getStaticDims()[1] : srcMemPtr->getStaticDims()[2];

    parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }

        auto* dst = pdst + dstIdx * feaDim * groupDim;
        auto* src = psrc + ii * feaDim * groupDim / 2;

        for (size_t g = 0; g < groupDim; g++) {
            // auto& deq_zp = zp[ii];
            // auto& deq_scale = scale[ii];
            auto& deq_zp = one_dim_zp ? zp[0] : zp[ii * groupDim + g];
            auto& deq_scale = scale[ii * groupDim + g];

            size_t k = 0;
            for (; k < feaDim; k += 2) {
                auto x = src[0];
                dst[0] = ((x & 0x0F) - deq_zp) * deq_scale;
                dst[1] = ((x >> 4) - deq_zp) * deq_scale;
                dst += 2;
                src++;
            }
            // Process last one if feaDim is odd
            for (; k < feaDim; k++) {
                auto x = src[0];
                dst[0] = ((x & 0x0F) - deq_zp) * deq_scale;
                dst++;
                src++;
            }
        }
    });
}

void GatherCompression::execReferenceU8() {
    DEBUG_LOG(getName(), "execReferenceU8");
    auto srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    auto idxMemPtr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    const auto* psrc = srcMemPtr->getDataAs<uint8_t>();
    const auto* pidx = idxMemPtr->getDataAs<int32_t>();

    bool one_dim_zp = getParentEdgeAt(GATHER_ZP)->getMemoryPtr()->getShape().getRank() == 1;
    const auto* zp = getSrcDataAtPortAs<float_t>(GATHER_ZP);
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    auto* pdst = getDstDataAtPortAs<float>(0);

    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto batch = idxDims[0];
    const auto seqLen = idxDims[1];

    auto axisDim = srcMemPtr->getStaticDims()[0];
    auto groupDim = srcMemPtr->getStaticDims().size() == 2 ? 1 : srcMemPtr->getStaticDims()[1];
    auto feaDim =
        srcMemPtr->getStaticDims().size() == 2 ? srcMemPtr->getStaticDims()[1] : srcMemPtr->getStaticDims()[2];


    parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }

        auto* src = psrc + ii * feaDim * groupDim;
        auto* dst = pdst + dstIdx * feaDim * groupDim;


        for (size_t g = 0; g < groupDim; g++) {
            auto& deq_zp = one_dim_zp ? zp[0] : zp[ii * groupDim + g];
            auto& deq_scale = scale[ii * groupDim + g];
            for (size_t k = 0; k < feaDim; k++) {
                dst[0] = (static_cast<float>(src[0]) - deq_zp) * deq_scale;
                dst++;
                src++;
            }
        }
    });
}

void GatherCompression::execReference() {
    auto srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    if (srcMemPtr->getPrecision() == ov::element::u8) {
        execReferenceU8();
    } else if (srcMemPtr->getPrecision() == ov::element::u4) {
        execReferenceU4();
    } else {
        THROW_ERROR("only support u4/u8 weights precision, don't support:", srcMemPtr->getPrecision());
    }
}

bool GatherCompression::created() const {
    return getType() == Type::GatherCompression;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov