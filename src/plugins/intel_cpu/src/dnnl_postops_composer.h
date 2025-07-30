// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling OneDNN attributes & post ops.
 * @file dnnl_postops_composer.h
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

enum class PostOpsMode : std::uint8_t {
    // Original mode using original post ops with modern zero points
    Original,
    // Legacy mode with fallback mechanism - try modern first, then legacy
    Legacy,
    // Forced legacy mode - directly use legacy post ops without trying original
    ForcedLegacy
};

// so far the API only support per-Tensor or per-OC
class DnnlPostOpsComposer {
public:
    DnnlPostOpsComposer(const PostOps& postOps,
                        const dnnl::engine& engine,
                        const VectorDims& outputDims,
                        size_t indexOfOutputChannelDim,
                        bool isINT8,
                        int weiScaleMaskPerChannel,
                        const MemoryArgs& memory,
                        dnnl::memory::data_type outDataType,
                        const std::vector<float>& legacyDqScales = {},
                        PostOpsMode postOpsMode = PostOpsMode::Original,
                        bool useLegacyZeroPoints = false,
                        dnnl::post_ops ops = dnnl::post_ops());
    DnnlPrimitiveAttrs compose();
    void appendDecompressionScales(const MemoryCPtr& scales_ptr,
                                   bool needTranspose,
                                   ov::element::Type dstPrecision,
                                   const VectorDims& weiDims);
    void appendDecompressionZeroPoints(const MemoryCPtr& zero_points_ptr,
                                       bool needTranspose,
                                       ov::element::Type dstPrecision,
                                       const VectorDims& weiDims);
    void appendDecompressionScalesLegacy(const MemoryCPtr& scales_ptr,
                                         bool needTranspose,
                                         ov::element::Type dstPrecision);
    void appendDecompressionZeroPointsLegacy(const MemoryCPtr& zero_points_ptr,
                                             bool needTranspose,
                                             ov::element::Type dstPrecision);
    void setDynamicQuantizationParams(uint64_t groupSize);

private:
    bool appendAttrPostOps(const ActivationPostOp& postOp, bool isLastPostOp, bool allowBinary = true);
    bool appendAttrPostOps(const ScaleShiftPostOp& postOp, bool isLastPostOp, bool allowBinary = true);
    bool appendAttrPostOps(const FakeQuantizePostOp& postOp,
                           bool isLastPostOp,
                           bool doRounding,
                           bool allowBinary = true);
    void appendAttrPostOpsLegacy(const ActivationPostOp& postOp);
    void appendAttrPostOpsLegacy(const ScaleShiftPostOp& postOp);
    void appendAttrPostOpsLegacy(const FakeQuantizePostOp& postOp);
    void appendBinary(dnnl::algorithm alg, const std::vector<float>& data);
    void appendEltwise(dnnl::algorithm alg, float alpha, float beta);
    void appendSum(float scale, int32_t zeroPoint, ov::element::Type dataType);
    void appendRoundHTE();
    bool appendScale(const std::vector<float>& scale, bool isLastPostOp, bool allowBinary = true);
    bool appendShift(const std::vector<float>& shift, bool allowBinary = true);
    bool appendLinear(const std::vector<float>& scale,
                      const std::vector<float>& shift,
                      bool isLastPostOp,
                      bool allowBinary = true);
    void appendClip(const std::vector<float>& low, const std::vector<float>& high);
    void appendDepthwiseConvolution(int inH,
                                    int inW,
                                    int kerH,
                                    int kerW,
                                    int strH,
                                    int strW,
                                    dnnl::memory::data_type inDataType);
    void appendZeroPoints(const MemoryArgs& memory);
    void appendZeroPointsLegacy(const MemoryArgs& memory);
    const dnnl::engine& engine;
    const PostOps& postOps;
    const VectorDims outputDims;
    size_t idxOC;
    const bool isINT8;  // only INT8 primitive support scales
    const int weightScaleMaskPerChannel;
    bool weightScaleAvailable = false;
    const dnnl::memory::data_type outDataType;
    const PostOpsMode postOpsMode;
    const bool useLegacyZeroPoints;

    dnnl::primitive_attr attr;
    MemoryArgs cpuArgs;
    dnnl_primitive_args dnnlArgs;

    VectorDims dimsPerTensor;
    VectorDims dimsPerOC;
    Dim OC;
    int wei_scale_mask = -1;
    std::vector<float> wei_scale_values;
    float dst_scale_val = 0.0F;
    dnnl::post_ops ops;

    void updateWeiScales();
    void updateDestScales();
};

}  // namespace ov::intel_cpu
