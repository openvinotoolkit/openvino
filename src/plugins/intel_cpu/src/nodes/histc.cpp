// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "histc.h"

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/histc.hpp"
#include "openvino/reference/histc.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool Histc::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto histc = ov::as_type_ptr<const ov::op::v17::Histc>(op);
        if (!histc) {
            errorMessage = "Only v17 Histc operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Histc::Histc(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto histc = ov::as_type_ptr<const ov::op::v17::Histc>(op);
    CPU_NODE_ASSERT(histc, "is not an instance of v17 Histc.");
    CPU_NODE_ASSERT(getOriginalInputsNumber() == 1 && getOriginalOutputsNumber() == 1,
                    "has incorrect number of input/output edges!");

    m_bins = histc->get_bins();
    m_min_val = histc->get_min_val();
    m_max_val = histc->get_max_val();
}

void Histc::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    m_data_precision = getOriginalInputPrecisionAtPort(INPUT_DATA_PORT);
    CPU_NODE_ASSERT(any_of(m_data_precision, ov::element::bf16, ov::element::f16, ov::element::f32, ov::element::f64),
                    "has unsupported input precision: ",
                    m_data_precision);

    m_output_precision = m_data_precision;
    addSupportedPrimDesc({{LayoutType::ncsp, m_data_precision}}, {{LayoutType::ncsp, m_output_precision}}, impl_desc_type::ref_any);
}

void Histc::prepareParams() {
    CPU_NODE_ASSERT(getSrcMemoryAtPort(INPUT_DATA_PORT), "has null input data memory.");
    CPU_NODE_ASSERT(getDstMemoryAtPort(OUTPUT_PORT), "has null output memory.");
}

void Histc::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto n = getSrcMemoryAtPort(INPUT_DATA_PORT)->getShape().getElementsCount();
    redefineOutputMemory({VectorDims{static_cast<size_t>(m_bins)}});

    switch (m_data_precision) {
    case ov::element::bf16:
        ov::reference::histc(getSrcDataAtPortAs<const ov::bfloat16>(INPUT_DATA_PORT),
                             n,
                             m_bins,
                             m_min_val,
                             m_max_val,
                             getDstDataAtPortAs<ov::bfloat16>(OUTPUT_PORT));
        return;
    case ov::element::f16:
        ov::reference::histc(getSrcDataAtPortAs<const ov::float16>(INPUT_DATA_PORT),
                             n,
                             m_bins,
                             m_min_val,
                             m_max_val,
                             getDstDataAtPortAs<ov::float16>(OUTPUT_PORT));
        return;
    case ov::element::f32:
        ov::reference::histc(getSrcDataAtPortAs<const float>(INPUT_DATA_PORT),
                             n,
                             m_bins,
                             m_min_val,
                             m_max_val,
                             getDstDataAtPortAs<float>(OUTPUT_PORT));
        return;
    case ov::element::f64:
        ov::reference::histc(getSrcDataAtPortAs<const double>(INPUT_DATA_PORT),
                             n,
                             m_bins,
                             m_min_val,
                             m_max_val,
                             getDstDataAtPortAs<double>(OUTPUT_PORT));
        return;
    default:
        CPU_NODE_THROW("has unsupported data precision: ", m_data_precision);
    }
}

bool Histc::created() const {
    return getType() == Type::Histc;
}

void Histc::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node
