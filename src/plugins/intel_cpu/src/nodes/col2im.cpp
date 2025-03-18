// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im.h"

#include "openvino/op/col2im.hpp"
#include "openvino/reference/col2im.hpp"

namespace ov::intel_cpu::node {
Col2Im::Col2Im(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto col2Im = ov::as_type_ptr<const ov::op::v15::Col2Im>(op);
    strides = col2Im->get_strides();
    dilations = col2Im->get_dilations();
    padsBegin = col2Im->get_pads_begin();
    padsEnd = col2Im->get_pads_end();
}

bool Col2Im::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::op::v15::Col2Im>(op)) {
            errorMessage = "Only opset15 Col2Im operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void Col2Im::getSupportedDescriptors() {
    // Validation is already done in the ov::opset15::Col2Im.
}

void Col2Im::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(0);
    addSupportedPrimDesc(
        {{LayoutType::ncsp, dataPrecision}, {LayoutType::ncsp, ov::element::i32}, {LayoutType::ncsp, ov::element::i32}},
        {{LayoutType::ncsp, dataPrecision}},
        impl_desc_type::ref);
}

bool Col2Im::created() const {
    return getType() == Type::Col2Im;
}

bool Col2Im::needPrepareParams() const {
    return false;
}

void Col2Im::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

template <class T, class T_idx>
void Col2Im::executeImpl() {
    ov::reference::col2im<T, T_idx>(getSrcDataAtPortAs<const T>(0),
                                    ov::Shape{getSrcMemoryAtPort(0)->getStaticDims()},
                                    getSrcDataAtPortAs<const T_idx>(1),
                                    getSrcDataAtPortAs<const T_idx>(2),
                                    getDstDataAtPortAs<T>(0),
                                    strides,
                                    dilations,
                                    padsBegin,
                                    padsEnd);
}

namespace {
struct Col2ImContext {
    Col2Im& node;
};
}  // namespace

template <typename T>
struct Col2Im::Col2ImExecute {
    using TData = typename std::tuple_element<0, T>::type;
    using TIndex = typename std::tuple_element<1, T>::type;

    void operator()(Col2ImContext& ctx) {
        ctx.node.executeImpl<TData, TIndex>();
    }
};
void Col2Im::execute(const dnnl::stream& strm) {
    auto dataPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto indexPrecision = getParentEdgeAt(1)->getMemory().getDesc().getPrecision();

    Col2ImContext ctx = {*this};

    OV_SWITCH(intel_cpu,
              Col2ImExecute,
              ctx,
              std::tie(dataPrecision, indexPrecision),
              OV_CASE2(ov::element::f32, ov::element::i32, float, int32_t),
              OV_CASE2(ov::element::f16, ov::element::i32, ov::float16, int32_t),
              OV_CASE2(ov::element::bf16, ov::element::i32, ov::bfloat16, int32_t),
              OV_CASE2(ov::element::i32, ov::element::i32, int32_t, int32_t),
              OV_CASE2(ov::element::i8, ov::element::i32, int8_t, int32_t),
              OV_CASE2(ov::element::u8, ov::element::i32, uint8_t, int32_t))
}
}  // namespace ov::intel_cpu::node
