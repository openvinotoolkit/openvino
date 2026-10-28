// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "histc.h"

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

namespace ov::intel_cpu::node {

Histc::Histc(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto histc = ov::as_type_ptr<const ov::op::v17::Histc>(op);
    m_bins = static_cast<size_t>(histc->get_bins());
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

    const auto prec = getOriginalInputPrecisionAtPort(0).is_real() ? ov::element::f32 : ov::element::i32;
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
                         m_bins,
                         m_min,
                         m_max);
}

void Histc::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (getParentEdgeAt(0)->getMemory().getDesc().getPrecision() == ov::element::f32) {
        executeImpl<float>();
    } else {
        executeImpl<int32_t>();
    }
}

}  // namespace ov::intel_cpu::node
