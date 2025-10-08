// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interpolate_common_executor.hpp"

#include <numeric>

#include "nodes/executors/common/ref_interpolate.hpp"

namespace ov::intel_cpu {

InterpolateCommonExecutor::InterpolateCommonExecutor(InterpolateAttrs attrs,
                                                     const MemoryArgs& /*memory*/,
                                                     const ExecutorContext::CPtr& context)
    : m_attrs(std::move(attrs)), m_context(context) {}

bool InterpolateCommonExecutor::update(const MemoryArgs& memory) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDesc();
    const auto& dstDesc = memory.at(ARG_DST)->getDesc();
    const auto& srcDims = srcDesc.getShape().getDims();
    const auto& dstDims = dstDesc.getShape().getDims();

    m_attrs.inPrc = srcDesc.getPrecision();
    m_attrs.outPrc = dstDesc.getPrecision();
    m_attrs.layout = dstDesc.hasLayoutType(LayoutType::ncsp) ? InterpolateLayoutType::planar : InterpolateLayoutType::by_channel;

    // compute scales
    std::vector<float> scales = m_attrs.dataScales;
    if (scales.empty()) {
        // use padded input dims for 'sizes' or read provided 'scales'
        std::vector<Dim> srcDimsPad = srcDims;
        if (m_attrs.hasPad && m_attrs.padBegin.size() == srcDims.size() && m_attrs.padEnd.size() == srcDims.size()) {
            for (size_t i = 0; i < srcDims.size(); ++i) {
                srcDimsPad[i] += static_cast<Dim>(m_attrs.padBegin[i] + m_attrs.padEnd[i]);
            }
        }
        std::vector<float> full(srcDims.size(), 1.0f);
        if (!m_attrs.axes.empty()) {
            if (m_attrs.shapeCalcMode == InterpolateShapeCalcMode::scales) {
                // v11 keeps scales at ARG_SRC_1, v4 keeps scales at ARG_SRC_2
                const float* scalesPtr = nullptr;
                if (auto it = memory.find(ARG_SRC_1); it != memory.end() &&
                    it->second->getDesc().getPrecision() == ov::element::f32) {
                    scalesPtr = it->second->getDataAs<const float>();
                } else if (auto it2 = memory.find(ARG_SRC_2); it2 != memory.end() &&
                           it2->second->getDesc().getPrecision() == ov::element::f32) {
                    scalesPtr = it2->second->getDataAs<const float>();
                }
                for (size_t i = 0; i < m_attrs.axes.size(); ++i) {
                    int axis = m_attrs.axes[i];
                    if (scalesPtr) full[axis] = scalesPtr[i];
                }
            } else {  // sizes
                const int32_t* sizesPtr = nullptr;
                size_t sizesElems = 0;
                if (auto it = memory.find(ARG_SRC_1); it != memory.end() &&
                    it->second->getDesc().getPrecision() == ov::element::i32) {
                    sizesPtr = it->second->getDataAs<const int32_t>();
                    sizesElems = it->second->getDesc().getShape().getElementsCount();
                }
                if (sizesPtr && sizesElems == srcDims.size()) {
                    // v4 TARGET_SHAPE has full rank
                    for (size_t i = 0; i < m_attrs.axes.size(); ++i) {
                        int axis = m_attrs.axes[i];
                        // For nearest mode, use padded dims only for asymmetric with pads (legacy border behavior)
                        bool use_padded = (m_attrs.mode == InterpolateMode::nearest) &&
                                          (m_attrs.coordTransMode == InterpolateCoordTransMode::asymmetric) &&
                                          m_attrs.hasPad;
                        const float denom = use_padded ? static_cast<float>(srcDimsPad[axis])
                                                        : ((m_attrs.mode == InterpolateMode::nearest)
                                                               ? static_cast<float>(srcDims[axis])
                                                               : static_cast<float>(srcDimsPad[axis]));
                        full[axis] = static_cast<float>(sizesPtr[axis]) / denom;
                    }
                } else {
                    // v11 SIZES provided per-axis in the same order as axes
                    for (size_t i = 0; i < m_attrs.axes.size(); ++i) {
                        int axis = m_attrs.axes[i];
                        const float denom_pad = static_cast<float>(srcDimsPad[axis]);
                        const float denom_nopad = static_cast<float>(srcDims[axis]);
                        bool use_padded = (m_attrs.mode == InterpolateMode::nearest) &&
                                          (m_attrs.coordTransMode == InterpolateCoordTransMode::asymmetric) &&
                                          m_attrs.hasPad;
                        const float denom = use_padded ? denom_pad
                                                       : ((m_attrs.mode == InterpolateMode::nearest) ? denom_nopad
                                                                                                     : denom_pad);
                        if (sizesPtr) full[axis] = static_cast<float>(sizesPtr[i]) / denom;
                        else full[axis] = static_cast<float>(dstDims[axis]) / denom;
                    }
                }
            }
        }
        scales = std::move(full);
    }

    // Map logical shapes/scales/pads to canonical 5D [N,C,D,H,W] regardless of memory layout.
    auto dimsTo5D = [&](const VectorDims& d) -> VectorDims {
        VectorDims out(5, 1);
        if (d.size() == 5) {
            out = d;
        } else if (d.size() == 4) {
            // Logical 4D is [N,C,H,W]
            out = {d[0], d[1], 1, d[2], d[3]};
        } else {
            out = to5Dim(d);
        }
        return out;
    };

    auto scalesTo5D = [&](const std::vector<float>& sc) {
        std::vector<float> out(5, 1.f);
        if (sc.size() == 5) {
            out = sc;
        } else if (sc.size() == 4) {
            // Ordered by [N,C,H,W]
            out = {sc[0], sc[1], 1.f, sc[2], sc[3]};
        } else if (sc.size() == 3) {
            // NCW -> N,C,D(=1),H(=1),W
            out = {sc[0], sc[1], 1.f, 1.f, sc[2]};
        } else {
            out = {1.f, 1.f, 1.f, 1.f, 1.f};
        }
        return out;
    };

    auto padsTo5D = [&](const std::vector<int>& v) {
        std::vector<int> out(5, 0);
        if (v.size() == 5) {
            out = v;
        } else if (v.size() == 4) {
            // Pads for logical [N,C,H,W]
            out = {v[0], v[1], 0, v[2], v[3]};
        } else if (v.size() == 3) {
            // NCW pads -> N,C,0,0,W
            out = {v[0], v[1], 0, 0, v[2]};
        }
        return out;
    };

    const auto src5D = dimsTo5D(srcDims);
    const auto dst5D = dimsTo5D(dstDims);
    const auto sc5D = scalesTo5D(scales);

    m_attrs.padBegin = padsTo5D(m_attrs.padBegin);
    m_attrs.padEnd = padsTo5D(m_attrs.padEnd);

    m_ref = std::make_shared<InterpolateRefExecutor>(m_attrs, src5D, dst5D, sc5D);
    return true;
}

void InterpolateCommonExecutor::execute(const MemoryArgs& memory) {
    auto src = memory.at(ARG_SRC);
    auto dst = memory.at(ARG_DST);
    m_ref->exec(src->getDataAs<const uint8_t>(), dst->getDataAs<uint8_t>(), nullptr);
}

}  // namespace ov::intel_cpu
