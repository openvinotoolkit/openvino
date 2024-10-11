// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling OneDNN attributes & post ops.
 * @file dnnl_postops_composer.h
 */
#pragma once

#include <dnnl_types.h>

#include "cpu_memory.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "post_ops.hpp"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"

namespace ov {
namespace intel_cpu {

// so far the API only support per-Tensor or per-OC
class DnnlPostOpsComposer {
public:
    DnnlPostOpsComposer(const PostOps& postOps,
                        const dnnl::engine& engine,
                        const VectorDims& outputDims,
                        const size_t indexOfOutputChannelDim,
                        const bool isINT8,
                        const int weiScaleMaskPerChannel,
                        const std::vector<float>& DQScales,
                        const bool hasBias,
                        const dnnl::memory::data_type outDataType);
    DnnlPrimitiveAttrs compose();
    void appendDecompressionScales(const MemoryCPtr& scales_ptr, bool needTranspose, ov::element::Type dstPrecision);
    void appendDecompressionZeroPoints(const MemoryCPtr& zero_points_ptr, bool needTranspose, ov::element::Type dstPrecision);
    void setDynamicQuantizationParams(uint64_t groupSize);

private:
    bool appendAttrPostOps(const ActivationPostOp& postOp, bool isLastPostOp, bool allowBinary = true);
    bool appendAttrPostOps(const ScaleShiftPostOp& postOp, bool isLastPostOp, bool allowBinary = true);
    bool appendAttrPostOps(const FakeQuantizePostOp& postOp,
                           bool isLastPostOp,
                           bool doRounding,
                           bool allowBinary = true);
    void appendBinary(const dnnl::algorithm alg, const std::vector<float>& data);
    void appendEltwise(const dnnl::algorithm alg, float alpha, float beta);
    void appendRoundHTE();
    bool appendScale(const std::vector<float>& scale, bool isLastPostOp, bool allowBinary = true);
    bool appendShift(const std::vector<float>& shift, bool allowBinary = true);
    bool appendLinear(const std::vector<float>& scale,
                      const std::vector<float>& shift,
                      bool isLastPostOp,
                      bool allowBinary = true);
    void appendClip(const std::vector<float>& low, const std::vector<float>& high);

    const dnnl::engine& engine;
    const PostOps& postOps;
    const VectorDims outputDims;
    size_t idxOC;
    const bool isINT8;  // only INT8 primitive support scales
    const int weightScaleMaskPerChannel;
    bool weightScaleAvailable = false;
    const dnnl::memory::data_type outDataType;

    dnnl::primitive_attr attr;
    MemoryArgs cpuArgs;
    dnnl_primitive_args dnnlArgs;

    VectorDims dimsPerTensor;
    VectorDims dimsPerOC;
    Dim OC;
    int wei_scale_mask = -1;
    std::vector<float> wei_scale_values;
    float dst_scale_val;
    dnnl::post_ops ops;

    void updateWeiScales();
    void updateDestScales();
};

}  // namespace intel_cpu
}  // namespace ov
