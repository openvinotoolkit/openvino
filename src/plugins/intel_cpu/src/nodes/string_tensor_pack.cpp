// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_pack.h"
#include "openvino/reference/string_tensor_pack.hpp"
#include "openvino/op/string_tensor_pack.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
StringTensorPack::StringTensorPack(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(0, 1))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
}

bool StringTensorPack::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v15::StringTensorPack>(op)) {
            errorMessage = "Only opset15 StringTensorPack operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void StringTensorPack::getSupportedDescriptors() {
    // Validation is already done in the ov::opset15::StringTensorPack
}

void StringTensorPack::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    ov::element::Type indicesPrecision = getOriginalInputPrecisionAtPort(0);
    addSupportedPrimDesc(
        {{LayoutType::ncsp, indicesPrecision}, {LayoutType::ncsp, indicesPrecision}, {LayoutType::ncsp, ov::element::u8}},
        {{LayoutType::ncsp, ov::element::string}},
        impl_desc_type::ref);
}

bool StringTensorPack::created() const {
    return getType() == Type::StringTensorPack;
}

bool StringTensorPack::needPrepareParams() const {
    return false;
}

void StringTensorPack::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <class T_idx>
void StringTensorPack::executeImpl() {
    const auto& data_shape = getSrcMemoryAtPort(0)->getStaticDims();
    ov::reference::string_tensor_pack(
        getSrcDataAtPortAs<const T_idx>(0),
        getSrcDataAtPortAs<const T_idx>(1),
        getSrcDataAtPortAs<const uint8_t>(2),
        getDstDataAtPortAs<std::string>(0),
        ov::shape_size(data_shape));
}

namespace {
struct StringTensorPackContext {
    StringTensorPack &node;
};
}

template<typename T_idx>
struct StringTensorPack::StringTensorPackExecute {
    void operator()(StringTensorPackContext& ctx) {
            ctx.node.executeImpl<T_idx>();
        }
};

void StringTensorPack::execute(dnnl::stream strm) {
    auto indicesPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    StringTensorPackContext ctx = {
            *this
    };
    OV_SWITCH(intel_cpu, StringTensorPackExecute, ctx, indicesPrecision,
              OV_CASE(ov::element::i32, int32_t),
              OV_CASE(ov::element::i64, int64_t))
}
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
