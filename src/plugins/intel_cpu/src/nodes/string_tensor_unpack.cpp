// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_unpack.h"
#include "openvino/reference/string_tensor_unpack.hpp"
#include "openvino/op/string_tensor_unpack.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
StringTensorUnpack::StringTensorUnpack(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool StringTensorUnpack::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v15::StringTensorUnpack>(op)) {
            errorMessage = "Only opset15 StringTensorUnpack operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void StringTensorUnpack::getSupportedDescriptors() {
    // Validation is already done in the ov::opset15::StringTensorUnpack
}

void StringTensorUnpack::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    addSupportedPrimDesc(
        {{LayoutType::ncsp, ov::element::string}},
        {{LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::u8}},
        impl_desc_type::ref);
}

bool StringTensorUnpack::created() const {
    return getType() == Type::StringTensorUnpack;
}

bool StringTensorUnpack::needPrepareParams() const {
    return false;
}

void StringTensorUnpack::executeDynamicImpl(dnnl::stream strm) {
    const auto& srcMemory = getSrcMemoryAtPort(0);
    const auto& srcDataDims = srcMemory->getStaticDims();
    const auto& srcData = srcMemory->getDataAs<std::string>();
    Dim stringCount = std::accumulate(srcDataDims.begin(), srcDataDims.end(), 1, std::multiplies<Dim>());
    size_t totalCharLength = 0;
    for (Dim i = 0; i < stringCount; ++i) {
        totalCharLength += srcData[i].length();
    }
    redefineOutputMemory({ srcDataDims, srcDataDims, {totalCharLength}});
    execute(strm);
}

void StringTensorUnpack::execute(dnnl::stream strm) {
    const auto stringCount = ov::shape_size(getSrcMemoryAtPort(0)->getStaticDims());
    ov::reference::string_tensor_unpack(
        getSrcDataAtPortAs<const std::string>(0),
        getDstDataAtPortAs<int32_t>(0),
        getDstDataAtPortAs<int32_t>(1),
        getDstDataAtPortAs<uint8_t>(2),
        stringCount);
}
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
