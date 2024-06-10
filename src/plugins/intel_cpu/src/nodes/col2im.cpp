// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im.h"
#include "common/cpu_convert.h"
#include "openvino/reference/col2im.hpp"

#include <openvino/opsets/opset15.hpp>

namespace ov {
namespace intel_cpu {
namespace node {
Col2Im::Col2Im(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    const auto col2Im = ov::as_type_ptr<const ov::opset15::Col2Im>(op);
    strides = col2Im->get_strides();
    dilations = col2Im->get_dilations();
    padsBegin = col2Im->get_pads_begin();
    padsEnd = col2Im->get_pads_end();
}

void Col2Im::getSupportedDescriptors() {
    // Validation is already done in the ov::opset15::Col2Im.
}

void Col2Im::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type dataPrecision = getOriginalInputPrecisionAtPort(0);
    ov::element::Type indexPrecision = getOriginalInputPrecisionAtPort(1);
    ov::element::Type outputPrecision = getOriginalOutputPrecisionAtPort(0);

    addSupportedPrimDesc(
        {{LayoutType::ncsp, dataPrecision}, {LayoutType::ncsp, indexPrecision}, {LayoutType::ncsp, indexPrecision}},
        {{LayoutType::ncsp, outputPrecision}},
        impl_desc_type::ref);
}

bool Col2Im::created() const {
    return getType() == Type::Col2Im;
}

bool Col2Im::needPrepareParams() const {
    return false;
}

void Col2Im::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <class T, class T_idx>
void Col2Im::executeImpl() {
    const auto indexPrecision = getSrcMemoryAtPort(1)->getPrecision();
    std::vector<T_idx> outputSizeVector(2);
    cpu_convert(getSrcMemoryAtPort(1)->getData(),
                outputSizeVector.data(),
                indexPrecision,
                ov::element::i64,
                2);

    std::vector<T_idx> kernelSizeVector(2);
    cpu_convert(getSrcMemoryAtPort(2)->getData(),
                kernelSizeVector.data(),
                indexPrecision,
                ov::element::i64,
                2);

    ov::reference::col2im<T, T_idx>(
        getSrcDataAtPortAs<const T>(0),
        ov::Shape{getSrcMemoryAtPort(0)->getStaticDims()},
        outputSizeVector.data(),
        kernelSizeVector.data(),
        getDstDataAtPortAs<T>(0),
        strides,
        dilations,
        padsBegin,
        padsEnd);
}

namespace {
struct Col2ImContext {
    Col2Im &node;
};
}

template<typename T>
struct Col2Im::Col2ImExecute {
    using TData = typename std::tuple_element<0, T>::type;
    using TIndex = typename std::tuple_element<1, T>::type;

    void operator()(Col2ImContext & ctx) {
            ctx.node.executeImpl<TData, TIndex>();
        }
};
void Col2Im::execute(dnnl::stream strm) {
    auto dataPrecision = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    auto indexPrecision = getParentEdgeAt(1)->getMemory().getDesc().getPrecision();

    Col2ImContext ctx = {
            *this
    };

    OV_SWITCH(intel_cpu, Col2ImExecute, ctx, std::tie(dataPrecision, indexPrecision),
              OV_CASE2(ov::element::f32, ov::element::i32, float, uint64_t),
              OV_CASE2(ov::element::f32, ov::element::i64, float, uint64_t),
              OV_CASE2(ov::element::f16, ov::element::i32, float, uint64_t),
              OV_CASE2(ov::element::f16, ov::element::i64, float, uint64_t),
              OV_CASE2(ov::element::i32, ov::element::i32, uint32_t, uint64_t),
              OV_CASE2(ov::element::i32, ov::element::i64, uint32_t, uint64_t))
}
}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
