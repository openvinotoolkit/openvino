// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_pad.hpp"

#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/Types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/acl/acl_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

using namespace arm_compute;

static PaddingMode getAclPaddingMode(PadMode padMode) {
    switch (padMode) {
    case CONSTANT:
        return PaddingMode::CONSTANT;
    case REFLECT:
        return PaddingMode::REFLECT;
    case SYMMETRIC:
        return PaddingMode::SYMMETRIC;
    default:
        OPENVINO_THROW("Unsupported pad mode for ACL Pad executor: ", static_cast<int>(padMode));
    }
}

static PaddingList getAclPaddingList(const PadAttrs& padAttrs, const MemoryDescPtr& srcDesc) {
    const auto rank = srcDesc->getShape().getRank();
    PaddingList padding(rank, std::make_pair(0U, 0U));

    for (size_t axis = 0; axis < rank; ++axis) {
        const auto aclAxis =
            axisCast(axis, rank, srcDesc->hasLayoutType(LayoutType::nspc) ? NHWC_TO_NCHW : NO_LAYOUT_CONVERSION);
        OPENVINO_ASSERT(aclAxis >= 0, "Layout conversion to ACL format has failed");
        padding[aclAxis] = {static_cast<unsigned int>(padAttrs.padsBegin[axis]),
                            static_cast<unsigned int>(padAttrs.padsEnd[axis])};
    }

    return padding;
}

AclPadExecutor::AclPadExecutor(ExecutorContext::CPtr context)
        : PadExecutor(std::move(context)),
            srcTensor(),
            dstTensor(),
            pad(nullptr) {}

bool AclPadExecutor::init(const PadAttrs& padAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          [[maybe_unused]] const dnnl::primitive_attr& attr) {
    this->padAttrs = padAttrs;

    auto srcShape = shapeCast(srcDescs[0]->getShape().getStaticDims());
    auto dstShape = shapeCast(dstDescs[0]->getShape().getStaticDims());
    if (srcDescs[0]->hasLayoutType(LayoutType::nspc) && dstDescs[0]->hasLayoutType(LayoutType::nspc)) {
        changeLayoutToNH_C({&srcShape, &dstShape});
    }

    const auto srcDataType = convertToQuantizedType(precisionToAclDataType(srcDescs[0]->getPrecision()));
    const auto dstDataType = convertToQuantizedType(precisionToAclDataType(dstDescs[0]->getPrecision()));

    TensorInfo srcTensorInfo(srcShape, 1, srcDataType, getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo(dstShape, 1, dstDataType, getAclDataLayoutByMemoryDesc(dstDescs[0]));

    if (any_of(srcDescs[0]->getPrecision(), ov::element::u8, ov::element::i8)) {
        srcTensorInfo.set_quantization_info(arm_compute::QuantizationInfo(1.0F));
    }
    if (any_of(dstDescs[0]->getPrecision(), ov::element::u8, ov::element::i8)) {
        dstTensorInfo.set_quantization_info(arm_compute::QuantizationInfo(1.0F));
    }

    const auto padding = getAclPaddingList(padAttrs, srcDescs[0]);
    const auto aclMode = getAclPaddingMode(padAttrs.padMode);
    const PixelValue padValue(padAttrs.padValue, srcTensorInfo.data_type(), srcTensorInfo.quantization_info());

    const auto validationStatus = NEPadLayer::validate(&srcTensorInfo, &dstTensorInfo, padding, padValue, aclMode);
    if (!validationStatus) {
        DEBUG_LOG("NEPadLayer validation failed: ", validationStatus.error_description());
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    pad = std::make_unique<NEPadLayer>();
    configureThreadSafe([&] {
        pad->configure(&srcTensor, &dstTensor, padding, padValue, aclMode);
    });

    return true;
}

void AclPadExecutor::exec(const std::vector<MemoryCPtr>& src,
                          const std::vector<MemoryPtr>& dst,
                          [[maybe_unused]] const void* post_ops_data_) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    pad->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
}

bool AclPadExecutorBuilder::isSupported(const PadAttrs& padAttrs,
                                        const std::vector<MemoryDescPtr>& srcDescs,
                                        const std::vector<MemoryDescPtr>& dstDescs) const {
    if (srcDescs.size() != 1U || dstDescs.size() != 1U) {
        DEBUG_LOG("AclPadExecutor expects one source and one destination tensor description");
        return false;
    }

    if (padAttrs.padsBegin.empty() || padAttrs.padsEnd.empty()) {
        DEBUG_LOG("AclPadExecutor supports static pads only");
        return false;
    }

    if (padAttrs.padMode == EDGE) {
        DEBUG_LOG("NEPadLayer does not support EDGE pad mode");
        return false;
    }

    if (srcDescs[0]->getPrecision() != dstDescs[0]->getPrecision() ||
        none_of(srcDescs[0]->getPrecision(),
                ov::element::f32,
                ov::element::i32,
                ov::element::bf16,
                ov::element::f16,
                ov::element::i8,
                ov::element::u8)) {
        DEBUG_LOG("NEPadLayer does not support precisions:",
                  " src[0]=",
                  srcDescs[0]->getPrecision(),
                  " dst[0]=",
                  dstDescs[0]->getPrecision());
        return false;
    }

    if (srcDescs[0]->getShape().getRank() > 4U) {
        DEBUG_LOG("NEPadLayer supports up to 4D tensors only. src[0] shape rank is ",
                  srcDescs[0]->getShape().getRank());
        return false;
    }

    if ((!srcDescs[0]->hasLayoutType(LayoutType::ncsp) || !dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
        (!srcDescs[0]->hasLayoutType(LayoutType::nspc) || !dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
        DEBUG_LOG("NEPadLayer does not support layouts:",
                  " src=",
                  srcDescs[0]->serializeFormat(),
                  " dst=",
                  dstDescs[0]->serializeFormat());
        return false;
    }

    if (std::any_of(padAttrs.padsBegin.begin(), padAttrs.padsBegin.end(), [](int32_t pad) {
            return pad < 0;
        }) ||
        std::any_of(padAttrs.padsEnd.begin(), padAttrs.padsEnd.end(), [](int32_t pad) {
            return pad < 0;
        })) {
        DEBUG_LOG("NEPadLayer does not support negative padding values");
        return false;
    }

    return true;
}

}  // namespace ov::intel_cpu