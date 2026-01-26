// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <tuple>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/col2im.hpp"
#include "openvino/reference/col2im.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"

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
    // 1. get data shape
    auto data_shape = getSrcMemoryAtPort(0)->getStaticDims();
    size_t data_rank = data_shape.size();

    // 2. get output_size
    auto output_size_mem = getSrcMemoryAtPort(1);
    const auto* output_size_ptr = output_size_mem->getDataAs<const int32_t>();

    // 3. get kernel_size
    auto kernel_size_mem = getSrcMemoryAtPort(2);
    const auto* kernel_size_ptr = kernel_size_mem->getDataAs<const int32_t>();

    // 4. calculate output_shape
    auto kernel_prod = static_cast<size_t>(kernel_size_ptr[0]) * static_cast<size_t>(kernel_size_ptr[1]);

    auto H = static_cast<size_t>(output_size_ptr[0]);
    auto W = static_cast<size_t>(output_size_ptr[1]);

    ov::Shape output_shape;
    if (data_rank == 2) {  // Case of Non-batched inputs
        size_t C = data_shape[0] / kernel_prod;
        output_shape = {C, H, W};
        redefineOutputMemory({output_shape});
        execute(strm);
    } else if (data_rank == 3) {  // Case of Batched inputs
        size_t N = data_shape[0];
        size_t C = data_shape[1] / kernel_prod;
        output_shape = {N, C, H, W};
        redefineOutputMemory({output_shape});
        execute(strm);
    } else {
        OPENVINO_THROW("Col2Im node supports only 2D(Non-Batched) or 3D(Batched) input tensors");
    }
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
void Col2Im::execute([[maybe_unused]] const dnnl::stream& strm) {
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
