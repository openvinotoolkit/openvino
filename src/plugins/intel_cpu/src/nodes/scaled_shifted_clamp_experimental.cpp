// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scaled_shifted_clamp_experimental.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <shape_inference/shape_inference_pass_through.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/scaled_shifted_clamp_experimental.hpp"

namespace ov::intel_cpu::node {

bool ScaledShiftedClamp::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                              std::string& errorMessage) noexcept {
    if (!ov::as_type_ptr<const ov::op::experimental::ScaledShiftedClamp>(op)) {
        errorMessage = "Only experimental::ScaledShiftedClamp op is supported";
        return false;
    }
    return true;
}

ScaledShiftedClamp::ScaledShiftedClamp(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto typed = ov::as_type_ptr<const ov::op::experimental::ScaledShiftedClamp>(op);
    m_scale = static_cast<float>(typed->get_scale());
    m_bias = static_cast<float>(typed->get_bias());
    m_lo = static_cast<float>(typed->get_lo());
    m_hi = static_cast<float>(typed->get_hi());
}

void ScaledShiftedClamp::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void ScaledShiftedClamp::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto* src = getSrcDataAtPortAs<const float>(0);
    auto* dst = getDstDataAtPortAs<float>(0);
    const auto n = getDstMemoryAtPort(0)->getShape().getElementsCount();
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::clamp(src[i] * m_scale + m_bias, m_lo, m_hi);
    }
}

}  // namespace ov::intel_cpu::node
