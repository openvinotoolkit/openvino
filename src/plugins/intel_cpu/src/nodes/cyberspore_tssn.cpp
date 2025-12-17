// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cyberspore_tssn.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/cyberspore_tssn.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/ternary.hpp"

namespace ov::intel_cpu::node {
namespace {
constexpr size_t INPUT_EVENTS_PORT = 0;
constexpr size_t STATE_PORT = 1;
constexpr size_t SELECTIVE_PORT = 2;
}

bool CybersporeTSSN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                           std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != ov::op::v0::CybersporeTSSN::get_type_info_static()) {
            errorMessage = "Only CybersporeTSSN operation from opset1 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

CybersporeTSSN::CybersporeTSSN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto cyberspore = ov::as_type_ptr<ov::op::v0::CybersporeTSSN>(op);
    m_homeostatic_setpoint = cyberspore->get_homeostatic_setpoint();
    m_decay_rate = cyberspore->get_decay_rate();
}

void CybersporeTSSN::getSupportedDescriptors() {
    if (getParentEdges().size() != 3) {
        CPU_NODE_THROW("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        CPU_NODE_THROW("has incorrect number of output edges.");
    }
}

void CybersporeTSSN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto selective_prc = getOriginalInputPrecisionAtPort(SELECTIVE_PORT);
    if (!selective_prc.is_real()) {
        selective_prc = ov::element::f32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::t2},
                          {LayoutType::ncsp, ov::element::t2},
                          {LayoutType::ncsp, selective_prc}},
                         {{LayoutType::ncsp, ov::element::t2}},
                         ref_any);
}

void CybersporeTSSN::prepareParams() {
    m_work_amount = getDstMemoryAtPort(0)->getShape().getElementsCount();
}

void CybersporeTSSN::execute(const dnnl::stream& /*strm*/) {
    if (m_work_amount == 0ULL) {
        return;
    }

    const auto& selective_memory = getParentEdgeAt(SELECTIVE_PORT)->getMemory();
    const auto selective_precision = selective_memory.getDesc().getPrecision();
    const auto selective_count = selective_memory.getShape().getElementsCount();
    CPU_NODE_ASSERT(selective_count == 1 || selective_count == m_work_amount,
                    "Selective parameters must be scalar or match the output shape. Got ",
                    selective_count,
                    " elements vs ",
                    m_work_amount,
                    ".");

    const void* selective_data = getSrcDataAtPort(SELECTIVE_PORT);
    CPU_NODE_ASSERT(selective_data != nullptr, "Selective parameters tensor has null data pointer.");

    if (selective_precision == ov::element::f32) {
        compute(reinterpret_cast<const float*>(selective_data), selective_count);
    } else if (selective_precision == ov::element::bf16) {
        compute(reinterpret_cast<const ov::bfloat16*>(selective_data), selective_count);
    } else if (selective_precision == ov::element::f16) {
        compute(reinterpret_cast<const ov::float16*>(selective_data), selective_count);
    } else {
        CPU_NODE_THROW("Selective parameters precision ",
                       selective_precision,
                       " is not supported by CybersporeTSSN CPU implementation.");
    }
}

void CybersporeTSSN::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool CybersporeTSSN::created() const {
    return getType() == Type::CybersporeTSSN;
}

template <typename SelectiveT>
void CybersporeTSSN::compute(const SelectiveT* selective_ptr, size_t selective_count) {
    const auto* event_ptr = reinterpret_cast<const uint8_t*>(getSrcDataAtPort(INPUT_EVENTS_PORT));
    const auto* state_ptr = reinterpret_cast<const uint8_t*>(getSrcDataAtPort(STATE_PORT));
    auto* dst_ptr = reinterpret_cast<uint8_t*>(getDstDataAtPort(0));

    CPU_NODE_ASSERT(event_ptr != nullptr && state_ptr != nullptr && dst_ptr != nullptr,
                    "CybersporeTSSN tensor pointers must not be null.");

    const bool broadcast_selective = selective_count == 1;
    const float setpoint = m_homeostatic_setpoint;
    const float decay = m_decay_rate;

    ov::parallel_for(m_work_amount, [&](size_t idx) {
        const auto event_raw = ternary::read(event_ptr, idx);
        const auto state_raw = ternary::read(state_ptr, idx);

        if (ternary::is_zero(event_raw) && ternary::is_zero(state_raw)) {
            ternary::write(dst_ptr, idx, 0U);
            return;
        }

        const float selective_value = static_cast<float>(selective_ptr[broadcast_selective ? 0 : idx]);
        const float selective_delta = selective_value - setpoint;
        const auto selective_bits = ternary::encode_from_float(selective_delta);

        const auto event_term = ternary::mul(event_raw, selective_bits);
        const auto decay_term = ternary::apply_decay(state_raw, decay);

        ternary::write(dst_ptr, idx, ternary::add(decay_term, event_term));
    });
}

// Explicit instantiations for supported types
template void CybersporeTSSN::compute<float>(const float*, size_t);
template void CybersporeTSSN::compute<ov::bfloat16>(const ov::bfloat16*, size_t);
template void CybersporeTSSN::compute<ov::float16>(const ov::float16*, size_t);

}  // namespace ov::intel_cpu::node
