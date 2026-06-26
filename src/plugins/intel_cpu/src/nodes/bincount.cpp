// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bincount.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/bincount.hpp"
#include "openvino/reference/bincount.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool Bincount::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto bc = ov::as_type_ptr<const ov::op::v17::Bincount>(op);
        if (!bc) {
            errorMessage = "Only v17 Bincount operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Bincount::Bincount(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto bc = ov::as_type_ptr<const ov::op::v17::Bincount>(op);
    CPU_NODE_ASSERT(bc, "is not an instance of v17 Bincount.");
    CPU_NODE_ASSERT(any_of(getOriginalInputsNumber(), 1U, 2U) && getOriginalOutputsNumber() == 1,
                    "has incorrect number of input/output edges!");

    m_minlength = bc->get_minlength();
    m_has_weights = getOriginalInputsNumber() == 2;
}

void Bincount::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    m_data_precision = getOriginalInputPrecisionAtPort(INPUT_DATA_PORT);
    CPU_NODE_ASSERT(any_of(m_data_precision,
                           ov::element::i32,
                           ov::element::i64,
                           ov::element::u8,
                           ov::element::u16,
                           ov::element::u32,
                           ov::element::u64),
                    "has unsupported input precision: ",
                    m_data_precision);

    if (m_has_weights) {
        m_weights_precision = getOriginalInputPrecisionAtPort(INPUT_WEIGHTS_PORT);
        CPU_NODE_ASSERT(any_of(m_weights_precision, ov::element::f32, ov::element::f64, ov::element::i32, ov::element::i64),
                        "has unsupported weights precision: ",
                        m_weights_precision);
        m_output_precision = m_weights_precision;
        addSupportedPrimDesc({{LayoutType::ncsp, m_data_precision}, {LayoutType::ncsp, m_weights_precision}},
                             {{LayoutType::ncsp, m_output_precision}},
                             impl_desc_type::ref_any);
    } else {
        m_output_precision = ov::element::i64;
        addSupportedPrimDesc({{LayoutType::ncsp, m_data_precision}},
                             {{LayoutType::ncsp, m_output_precision}},
                             impl_desc_type::ref_any);
    }
}

void Bincount::prepareParams() {
    CPU_NODE_ASSERT(getSrcMemoryAtPort(INPUT_DATA_PORT), "has null input data memory.");
    CPU_NODE_ASSERT(getDstMemoryAtPort(OUTPUT_PORT), "has null output memory.");
    if (m_has_weights) {
        CPU_NODE_ASSERT(getSrcMemoryAtPort(INPUT_WEIGHTS_PORT), "has null weights memory.");
    }
}

size_t Bincount::get_output_size() const {
    const auto n = getSrcMemoryAtPort(INPUT_DATA_PORT)->getShape().getElementsCount();
    switch (m_data_precision) {
    case ov::element::i32:
        return ov::reference::bincount_output_size(getSrcDataAtPortAs<const int32_t>(INPUT_DATA_PORT), n, m_minlength);
    case ov::element::i64:
        return ov::reference::bincount_output_size(getSrcDataAtPortAs<const int64_t>(INPUT_DATA_PORT), n, m_minlength);
    case ov::element::u8:
        return ov::reference::bincount_output_size(getSrcDataAtPortAs<const uint8_t>(INPUT_DATA_PORT), n, m_minlength);
    case ov::element::u16:
        return ov::reference::bincount_output_size(getSrcDataAtPortAs<const uint16_t>(INPUT_DATA_PORT), n, m_minlength);
    case ov::element::u32:
        return ov::reference::bincount_output_size(getSrcDataAtPortAs<const uint32_t>(INPUT_DATA_PORT), n, m_minlength);
    case ov::element::u64:
        return ov::reference::bincount_output_size(getSrcDataAtPortAs<const uint64_t>(INPUT_DATA_PORT), n, m_minlength);
    default:
        CPU_NODE_THROW("has unsupported data precision: ", m_data_precision);
    }
}

void Bincount::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto n = getSrcMemoryAtPort(INPUT_DATA_PORT)->getShape().getElementsCount();
    const auto out_size = get_output_size();
    redefineOutputMemory({VectorDims{out_size}});

    if (!m_has_weights) {
        auto* dst = getDstDataAtPortAs<int64_t>(OUTPUT_PORT);
        switch (m_data_precision) {
        case ov::element::i32:
            ov::reference::bincount(getSrcDataAtPortAs<const int32_t>(INPUT_DATA_PORT), n, m_minlength, dst, out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount(getSrcDataAtPortAs<const int64_t>(INPUT_DATA_PORT), n, m_minlength, dst, out_size);
            return;
        case ov::element::u8:
            ov::reference::bincount(getSrcDataAtPortAs<const uint8_t>(INPUT_DATA_PORT), n, m_minlength, dst, out_size);
            return;
        case ov::element::u16:
            ov::reference::bincount(getSrcDataAtPortAs<const uint16_t>(INPUT_DATA_PORT), n, m_minlength, dst, out_size);
            return;
        case ov::element::u32:
            ov::reference::bincount(getSrcDataAtPortAs<const uint32_t>(INPUT_DATA_PORT), n, m_minlength, dst, out_size);
            return;
        case ov::element::u64:
            ov::reference::bincount(getSrcDataAtPortAs<const uint64_t>(INPUT_DATA_PORT), n, m_minlength, dst, out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported data precision: ", m_data_precision);
        }
    }

    CPU_NODE_ASSERT(n == getSrcMemoryAtPort(INPUT_WEIGHTS_PORT)->getShape().getElementsCount(),
                    "requires data and weights to have the same number of elements.");

    switch (m_data_precision) {
    case ov::element::i32: {
        const auto* data = getSrcDataAtPortAs<const int32_t>(INPUT_DATA_PORT);
        switch (m_weights_precision) {
        case ov::element::f32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const float>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<float>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::f64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const double>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<double>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int32_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int32_t>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int64_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int64_t>(OUTPUT_PORT),
                                             out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported weights precision: ", m_weights_precision);
        }
    }
    case ov::element::i64: {
        const auto* data = getSrcDataAtPortAs<const int64_t>(INPUT_DATA_PORT);
        switch (m_weights_precision) {
        case ov::element::f32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const float>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<float>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::f64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const double>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<double>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int32_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int32_t>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int64_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int64_t>(OUTPUT_PORT),
                                             out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported weights precision: ", m_weights_precision);
        }
    }
    case ov::element::u8: {
        const auto* data = getSrcDataAtPortAs<const uint8_t>(INPUT_DATA_PORT);
        switch (m_weights_precision) {
        case ov::element::f32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const float>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<float>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::f64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const double>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<double>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int32_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int32_t>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int64_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int64_t>(OUTPUT_PORT),
                                             out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported weights precision: ", m_weights_precision);
        }
    }
    case ov::element::u16: {
        const auto* data = getSrcDataAtPortAs<const uint16_t>(INPUT_DATA_PORT);
        switch (m_weights_precision) {
        case ov::element::f32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const float>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<float>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::f64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const double>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<double>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int32_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int32_t>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int64_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int64_t>(OUTPUT_PORT),
                                             out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported weights precision: ", m_weights_precision);
        }
    }
    case ov::element::u32: {
        const auto* data = getSrcDataAtPortAs<const uint32_t>(INPUT_DATA_PORT);
        switch (m_weights_precision) {
        case ov::element::f32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const float>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<float>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::f64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const double>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<double>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int32_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int32_t>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int64_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int64_t>(OUTPUT_PORT),
                                             out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported weights precision: ", m_weights_precision);
        }
    }
    case ov::element::u64: {
        const auto* data = getSrcDataAtPortAs<const uint64_t>(INPUT_DATA_PORT);
        switch (m_weights_precision) {
        case ov::element::f32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const float>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<float>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::f64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const double>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<double>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i32:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int32_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int32_t>(OUTPUT_PORT),
                                             out_size);
            return;
        case ov::element::i64:
            ov::reference::bincount_weighted(data,
                                             getSrcDataAtPortAs<const int64_t>(INPUT_WEIGHTS_PORT),
                                             n,
                                             m_minlength,
                                             getDstDataAtPortAs<int64_t>(OUTPUT_PORT),
                                             out_size);
            return;
        default:
            CPU_NODE_THROW("has unsupported weights precision: ", m_weights_precision);
        }
    }
    default:
        CPU_NODE_THROW("has unsupported data precision: ", m_data_precision);
    }
}

void Bincount::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool Bincount::created() const {
    return getType() == Type::Bincount;
}

}  // namespace ov::intel_cpu::node
