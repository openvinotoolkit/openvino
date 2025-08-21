// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"

static constexpr int MAX_INPUT_INTERPOLATE = 8;

namespace ov::intel_cpu {

inline VectorDims getPaddedInputShape(const VectorDims& srcDims,
                                      const std::vector<int>& padBegin,
                                      const std::vector<int>& padEnd) {
    VectorDims paddedShape;
    int dataRank = srcDims.size();
    for (int i = 0; i < dataRank; i++) {
        paddedShape.push_back(srcDims[i] + padBegin[i] + padEnd[i]);
    }
    return paddedShape;
}

inline int clipCoord(int pos, int length) {
    return std::max(0, std::min(pos, length - 1));
}

inline size_t getSpatialDimsNum(const Dim rank) {
    switch (rank) {
    case 1:
    case 3:
        return 1;
    case 2:
    case 4:
        return 2;
    case 5:
        return 3;
    default:
        OPENVINO_THROW("Can't define number spatial");
    }
}

// w/hw/ncw/nchw/ncdhw to ncdhw
inline VectorDims to5Dim(VectorDims casesDim) {
    size_t caseSize = casesDim.size();
    VectorDims dim5(5, 1LU);
    dim5[4] = casesDim[caseSize - 1];
    if (caseSize > 1) {
        dim5[3] = casesDim[caseSize - 2];
    }
    if (caseSize > 2) {
        dim5[0] = casesDim[0];
    }
    if (caseSize > 3) {
        dim5[1] = casesDim[1];
    }
    if (caseSize > 4) {
        dim5[2] = casesDim[2];
    }
    if (caseSize == 3) {  // nhw -> ncw
        dim5[1] = dim5[3];
        dim5[3] = 1LU;
    }
    return dim5;
}

static inline float triangleCoeff(float x) {
    return (std::max)(0.0F, 1 - std::abs(x));
}

class InterpolateExecutor : public Executor {
public:
    static constexpr size_t DATA_ID = 0;
    static constexpr size_t TARGET_SHAPE_ID = 1;
    static constexpr size_t SCALES_ID = 2;
    static constexpr size_t AXES_ID = 3;
    static constexpr int CUBIC_GRID_LEN = 4;
    explicit InterpolateExecutor(ExecutorContext::CPtr context) : _context(std::move(context)) {}

    virtual bool init(const InterpolateAttrs& interpolateAttrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr& attr);
    virtual void exec(const std::vector<MemoryCPtr>& src,
                      const std::vector<MemoryPtr>& dst,
                      const void* post_ops_data_) = 0;
    
    // Bring base class exec into scope to avoid hiding
    using Executor::exec;
    [[nodiscard]] virtual impl_desc_type getImplType() const = 0;
    
    // Executor interface
    void execute(const MemoryArgs& memory) override {
        std::vector<MemoryCPtr> srcMemory;
        std::vector<MemoryPtr> dstMemory;
        for (const auto& [k, v] : memory) {
            if (k == ARG_DST) {
                dstMemory.push_back(v);
            } else {
                srcMemory.push_back(std::const_pointer_cast<const IMemory>(v));
            }
        }
        exec(srcMemory, dstMemory, nullptr);
    }
    
    [[nodiscard]] impl_desc_type implType() const override {
        return getImplType();
    }

    virtual ~InterpolateExecutor() = default;
    [[nodiscard]] VectorDims getSrcDimPad5d() const {
        return srcDimPad5d;
    }
    const uint8_t* padPreprocess(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst);

private:
    void buildTblNN(const VectorDims& srcDimPad5d,
                    const VectorDims& dstDim5d,
                    const std::vector<float>& dataScales,
                    InterpolateLayoutType layout,
                    InterpolateNearestMode nearestMode);
    void buildTblLinearOnnx(const VectorDims& srcDimPad5d,
                            const VectorDims& dstDim5d,
                            const std::vector<float>& dataScales,
                            InterpolateLayoutType layout);
    void buildTblLinear(const VectorDims& srcDimPad5d,
                        const VectorDims& dstDim5d,
                        const std::vector<float>& dataScales,
                        int kernel_width,
                        bool antialias);
    void buildTblCubic(const VectorDims& srcDimPad5d,
                       const VectorDims& dstDim5d,
                       const std::vector<float>& dataScales,
                       float cubicCoeff,
                       InterpolateLayoutType layout);

    [[nodiscard]] float coordTransToInput(int outCoord, float scale, int inShape, int outShape) const;
    [[nodiscard]] static int nearestRound(float origin, bool isDownsample, InterpolateNearestMode nearestMode);
    void linearOnnxCF(int outCoord,
                      float scale,
                      int inShape,
                      int outShape,
                      int& index0,
                      int& index1,
                      float& weight0,
                      float& weight1);
    static std::vector<float> getCubicCoeffs(float mantissa, float a);

protected:
    InterpolateAttrs interpAttrs;
    VectorDims srcDimPad5d, dstDim5d;
    size_t srcDataSize = 0UL, dstDataSize = 0UL;
    int spatialDimSize = 0;
    size_t dataRank = 0UL;
    std::vector<int> indexTable;
    const ExecutorContext::CPtr _context;
};

using InterpolateExecutorPtr = std::shared_ptr<InterpolateExecutor>;
using InterpolateExecutorCPtr = std::shared_ptr<const InterpolateExecutor>;

class InterpolateExecutorBuilder {
public:
    virtual ~InterpolateExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const InterpolateAttrs& InterpolateAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual InterpolateExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using InterpolateExecutorBuilderPtr = std::shared_ptr<InterpolateExecutorBuilder>;
using InterpolateExecutorBuilderCPtr = std::shared_ptr<const InterpolateExecutorBuilder>;
}  // namespace ov::intel_cpu
