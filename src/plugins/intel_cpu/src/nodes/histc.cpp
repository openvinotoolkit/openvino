// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "histc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/histc.hpp"
#include "openvino/reference/histc.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

Histc::Histc(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto histc = ov::as_type_ptr<const ov::op::v17::Histc>(op);
    m_min = histc->get_min();
    m_max = histc->get_max();
}

bool Histc::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    if (!ov::is_type<ov::op::v17::Histc>(op)) {
        errorMessage = "Only opset17 Histc operation is supported";
        return false;
    }
    return true;
}

void Histc::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto prec = getOriginalInputPrecisionAtPort(0);
    if (prec.is_real()) {
        prec = ov::element::f32;
    } else if (none_of(prec, ov::element::i8, ov::element::u8)) {
        prec = ov::element::i32;
    }
    addSupportedPrimDesc({{LayoutType::ncsp, prec}}, {{LayoutType::ncsp, prec}}, impl_desc_type::ref);
}

bool Histc::created() const {
    return getType() == Type::Histc;
}

template <class T>
void Histc::executeImpl() {
    const auto& src = getSrcMemoryAtPort(0);
    ov::reference::histc(src->getDataAs<const T>(),
                         getDstDataAtPortAs<T>(0),
                         ov::shape_size(src->getStaticDims()),
                         ov::shape_size(getDstMemoryAtPort(0)->getStaticDims()),
                         m_min,
                         m_max);
}

void Histc::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto prec = getParentEdgeAt(0)->getMemory().getDesc().getPrecision();
    switch (prec) {
    case ov::element::f32:
        executeImpl<float>();
        break;
    case ov::element::i32:
        executeImpl<int32_t>();
        break;
    case ov::element::i8:
        executeImpl<int8_t>();
        break;
    case ov::element::u8:
        executeImpl<uint8_t>();
        break;
    default:
        CPU_NODE_THROW("has unsupported precision: ", prec);
    }
}

}  // namespace ov::intel_cpu::node
