// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bevpool_v2.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_parallel.hpp"
#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/bevpool_v2.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

bool BevPoolV2::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != ov::op::v15::BevPoolV2::get_type_info_static()) {
            errorMessage = "Only BevPoolV2 operation from opset15 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

BevPoolV2::BevPoolV2(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto bevpool = std::dynamic_pointer_cast<const ov::op::v15::BevPoolV2>(op);
    m_input_channels = bevpool->get_input_channels();
    m_output_channels = bevpool->get_output_channels();
    m_image_width = bevpool->get_image_width();
    m_image_height = bevpool->get_image_height();
    m_feature_width = bevpool->get_feature_width();
    m_feature_height = bevpool->get_feature_height();
    m_d_bound = bevpool->get_d_bound();

    m_feature_area = static_cast<int64_t>(m_image_width) * static_cast<int64_t>(m_image_height);
    m_depth_bins = static_cast<int64_t>((m_d_bound.max - m_d_bound.min) / m_d_bound.step);
    m_depth_span = m_depth_bins * m_feature_area;
    m_out_plane = static_cast<int64_t>(m_feature_width) * static_cast<int64_t>(m_feature_height);
}

void BevPoolV2::getSupportedDescriptors() {
    if (getParentEdges().size() != 4) {
        CPU_NODE_THROW("has incorrect number of input edges: ", getParentEdges().size());
    }
    if (getChildEdges().empty()) {
        CPU_NODE_THROW("has incorrect number of output edges.");
    }
}

void BevPoolV2::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto cf_prc = getOriginalInputPrecisionAtPort(CF_IDX);
    auto dw_prc = getOriginalInputPrecisionAtPort(DW_IDX);
    auto idx_prc = getOriginalInputPrecisionAtPort(IDX_IDX);
    auto itv_prc = getOriginalInputPrecisionAtPort(ITV_IDX);

    // Feature inputs: support f32 and f16
    if (cf_prc != ov::element::f32 && cf_prc != ov::element::f16) {
        cf_prc = ov::element::f32;
    }
    if (dw_prc != ov::element::f32 && dw_prc != ov::element::f16) {
        dw_prc = ov::element::f32;
    }

    // Index inputs: support i32, i64, u32 and u64
    const auto is_supported_index_prc = [](const ov::element::Type& prc) {
        return prc == ov::element::i32 || prc == ov::element::i64 || prc == ov::element::u32 || prc == ov::element::u64;
    };
    if (!is_supported_index_prc(idx_prc)) {
        idx_prc = ov::element::i32;
    }
    if (!is_supported_index_prc(itv_prc)) {
        itv_prc = ov::element::i32;
    }

    auto out_prc = cf_prc;

    addSupportedPrimDesc({{LayoutType::ncsp, cf_prc},
                          {LayoutType::ncsp, dw_prc},
                          {LayoutType::ncsp, idx_prc},
                          {LayoutType::ncsp, itv_prc}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

template <typename T, typename IdxT, typename ItvT>
void BevPoolV2::executeImpl() {
    const auto& cpu_parallel = context->getCpuParallel();

    const auto* cf_data = getSrcDataAtPortAs<const T>(CF_IDX);
    const auto* dw_data = getSrcDataAtPortAs<const T>(DW_IDX);
    const auto* idx_data = getSrcDataAtPortAs<const IdxT>(IDX_IDX);
    const auto* itv_data = getSrcDataAtPortAs<const ItvT>(ITV_IDX);
    auto* out_data = getDstDataAtPortAs<T>(0);

    const auto itv_len = static_cast<int64_t>(getSrcMemoryAtPort(ITV_IDX)->getShape().getElementsCount());
    CPU_NODE_ASSERT(itv_len % 3 == 0, "Intervals input length must be divisible by 3. Got ", itv_len);
    const auto interval_count = itv_len / 3;

    const auto idx_len = static_cast<int64_t>(getSrcMemoryAtPort(IDX_IDX)->getShape().getElementsCount());
    const auto dw_len = static_cast<int64_t>(getSrcMemoryAtPort(DW_IDX)->getShape().getElementsCount());
    const auto cf_len = static_cast<int64_t>(getSrcMemoryAtPort(CF_IDX)->getShape().getElementsCount());
    const auto out_len = static_cast<int64_t>(getDstMemoryAtPort(0)->getShape().getElementsCount());

    std::memset(out_data, 0, out_len * sizeof(T));

    cpu_parallel->parallel_for(interval_count, [&](int64_t interval) {
        const auto start = static_cast<int64_t>(itv_data[interval * 3 + 0]);
        const auto end = static_cast<int64_t>(itv_data[interval * 3 + 1]);
        const auto bev_base = static_cast<int64_t>(itv_data[interval * 3 + 2]);

        if (start < 0 || end < start || end > idx_len) {
            return;
        }

        for (uint32_t c = 0; c < m_output_channels; ++c) {
            float acc = 0.F;
            for (int64_t i = start; i < end; ++i) {
                const auto dw_index = static_cast<int64_t>(idx_data[i]);
                if (dw_index < 0 || dw_index >= dw_len) {
                    continue;
                }

                const auto camera_idx = dw_index / m_depth_span;
                const auto feature_idx = dw_index % m_feature_area;
                const auto cf_offset =
                    (camera_idx * m_feature_area + feature_idx) * static_cast<int64_t>(m_input_channels) +
                    static_cast<int64_t>(c);

                if (cf_offset < 0 || cf_offset >= cf_len) {
                    continue;
                }

                acc += static_cast<float>(cf_data[cf_offset]) * static_cast<float>(dw_data[dw_index]);
            }

            const auto out_index = bev_base + static_cast<int64_t>(c) * m_out_plane;
            if (out_index >= 0 && out_index < out_len) {
                out_data[out_index] = static_cast<T>(acc);
            }
        }
    });
}

template <typename T, typename IdxT>
void BevPoolV2::dispatchIntervalType(const ov::element::Type& itv_prc) {
    if (itv_prc == ov::element::i64) {
        executeImpl<T, IdxT, int64_t>();
    } else if (itv_prc == ov::element::u32) {
        executeImpl<T, IdxT, uint32_t>();
    } else if (itv_prc == ov::element::u64) {
        executeImpl<T, IdxT, uint64_t>();
    } else {
        executeImpl<T, IdxT, int32_t>();
    }
}

template <typename T>
void BevPoolV2::dispatchIndexType(const ov::element::Type& idx_prc, const ov::element::Type& itv_prc) {
    if (idx_prc == ov::element::i64) {
        dispatchIntervalType<T, int64_t>(itv_prc);
    } else if (idx_prc == ov::element::u32) {
        dispatchIntervalType<T, uint32_t>(itv_prc);
    } else if (idx_prc == ov::element::u64) {
        dispatchIntervalType<T, uint64_t>(itv_prc);
    } else {
        dispatchIntervalType<T, int32_t>(itv_prc);
    }
}

void BevPoolV2::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto cf_prc = getSrcMemoryAtPort(CF_IDX)->getDesc().getPrecision();
    const auto idx_prc = getSrcMemoryAtPort(IDX_IDX)->getDesc().getPrecision();
    const auto itv_prc = getSrcMemoryAtPort(ITV_IDX)->getDesc().getPrecision();

    if (cf_prc == ov::element::f16) {
        dispatchIndexType<ov::float16>(idx_prc, itv_prc);
    } else {
        dispatchIndexType<float>(idx_prc, itv_prc);
    }
}

void BevPoolV2::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool BevPoolV2::created() const {
    return getType() == Type::BevPoolV2;
}

}  // namespace ov::intel_cpu::node
