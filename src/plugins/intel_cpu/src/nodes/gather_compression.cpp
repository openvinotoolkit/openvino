// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_compression.h"

#include <openvino/op/gather.hpp>
#include <openvino/op/constant.hpp>

#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include <openvino/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include "common/cpu_convert.h"
#include "utils/general_utils.h"
#include "kernels/x64/gather_uni_kernel.hpp"
#include <partitioned_mem_mgr.h>
#include "shape_inference/custom/gather.hpp"
#include "utils/ngraph_utils.hpp"
#include "transformations/cpu_opset/common/op/gather_compression.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace dnnl::impl::cpu;

#define THROW_ERROR(...) OPENVINO_THROW(getTypeStr(), " node with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {
namespace node {

bool GatherCompression::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto gather_compression = std::dynamic_pointer_cast<const GatherCompressionNode>(op);
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

    if (op->get_input_size() != 5 || op->get_output_size() != 1)
        THROW_ERROR("has incorrect number of input/output edges!");

    isAxisInputConst = ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(GATHER_AXIS));
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
                          {LayoutType::ncsp, ov::element::i32},
                          {LayoutType::ncsp, ov::element::i32, isAxisInputConst}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         ref_any);
}

bool GatherCompression::needPrepareParams() const {
    return inputShapesModified();
}

void GatherCompression::prepareParams() {
    auto dataMemPtr = getSrcMemoryAtPort(GATHER_DATA);
    if (!dataMemPtr || !dataMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input data memory.");

    auto zpMemPtr = getSrcMemoryAtPort(GATHER_ZP);
    if (!zpMemPtr || !zpMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input ZP memory.");

    auto scaleMemPtr = getSrcMemoryAtPort(GATHER_SCALE);
    if (!scaleMemPtr || !scaleMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input scale memory.");

    auto idxMemPtr = getSrcMemoryAtPort(GATHER_INDICES);
    if (!idxMemPtr || !idxMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input indices memory.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_ERROR(" has unidentified preferable primitive descriptor.");

    auto axisMemPtr = getSrcMemoryAtPort(GATHER_AXIS);
    if (!axisMemPtr || !axisMemPtr->isAllocated())
        THROW_ERROR(" has not allocated input axis memory.");
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

    const auto* zp = getSrcDataAtPortAs<float_t>(GATHER_ZP);
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    auto* pdst = getDstDataAtPortAs<float>(0);

    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto batch = idxDims[0];
    const auto seqLen = idxDims[1];

    auto axisDim = srcMemPtr->getStaticDims()[0];
    auto feaDim = srcMemPtr->getStaticDims()[1];

    parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }

        auto get_u4_value = [&](int32_t indx) {
            int ii_indx = indx >> 1;
            int ii_offset = indx % 2;
            if (ii_offset) {
                return static_cast<float>(psrc[ii_indx] >> 4);
            } else {
                return static_cast<float>(psrc[ii_indx] & 0Xf);
            }
        };

        int32_t offset = ii * feaDim;
        auto* dst = pdst + dstIdx * feaDim;
        auto& deq_zp = zp[ii];
        auto& deq_scale = scale[ii];
        for (size_t k = 0; k < feaDim; k++) {
            dst[k] = (get_u4_value(offset + k) - deq_zp) * deq_scale;
        }
    });
}

void GatherCompression::execReferenceU8() {
    DEBUG_LOG(getName(), "execReferenceU8");
    auto srcMemPtr = getParentEdgeAt(GATHER_DATA)->getMemoryPtr();
    auto idxMemPtr = getParentEdgeAt(GATHER_INDICES)->getMemoryPtr();
    const auto* psrc = srcMemPtr->getDataAs<uint8_t>();
    const auto* pidx = idxMemPtr->getDataAs<int32_t>();

    const auto* zp = getSrcDataAtPortAs<float_t>(GATHER_ZP);
    const auto* scale = getSrcDataAtPortAs<float_t>(GATHER_SCALE);
    auto* pdst = getDstDataAtPortAs<float>(0);

    const auto& idxDims = idxMemPtr->getStaticDims();
    const auto batch = idxDims[0];
    const auto seqLen = idxDims[1];

    auto axisDim = srcMemPtr->getStaticDims()[0];
    auto feaDim = srcMemPtr->getStaticDims()[1];

    parallel_for2d(batch, seqLen, [&](size_t b, size_t s) {
        auto dstIdx = b * seqLen + s;
        auto ii = pidx[dstIdx];
        if (ii < 0) {
            if (reverseIndexing)
                ii += axisDim;
            else
                ii = axisDim;
        }

        auto* src = psrc + ii * feaDim;
        auto* dst = pdst + dstIdx * feaDim;
        auto& deq_zp = zp[ii];
        auto& deq_scale = scale[ii];
        for (size_t k = 0; k < feaDim; k++) {
            dst[k] = (static_cast<float>(src[k]) - deq_zp) * deq_scale;
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
