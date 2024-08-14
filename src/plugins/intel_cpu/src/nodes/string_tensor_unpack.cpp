// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string_tensor_unpack.h"
#include "openvino/reference/string_tensor_unpack.hpp"
#include "openvino/op/string_tensor_unpack.hpp"

namespace ov {
namespace intel_cpu {
namespace node {
StringTensorUnpack::StringTensorUnpack(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(0))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto stringTensorUnpack = ov::as_type_ptr<const ov::op::v15::StringTensorUnpack>(op);
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
    execute(strm);
}

void StringTensorUnpack::executeImpl() {
    const auto string_count = ov::shape_size(getSrcMemoryAtPort(0)->getStaticDims());
    ov::reference::string_tensor_unpack(
        getSrcDataAtPortAs<const std::string>(0),
        getDstDataAtPortAs<int32_t>(0),
        getDstDataAtPortAs<int32_t>(1),
        getDstDataAtPortAs<uint8_t>(2),
        string_count);
}

namespace {
struct StringTensorUnpackContext {
    StringTensorUnpack &node;
};
}

struct StringTensorUnpack::StringTensorUnpackExecute {
    void operator()(StringTensorUnpackContext & ctx) {
            ctx.node.executeImpl();
        }
};
void StringTensorUnpack::execute(dnnl::stream strm) {
    StringTensorUnpackContext ctx = {
            *this
    };
    StringTensorUnpackExecute()(ctx);
}
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
