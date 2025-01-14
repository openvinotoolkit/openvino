// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Convinience wrapper class for handling OneDNN attributes & post ops.
 * @file dnnl_postops_composer.h
 */
#pragma once

#include <string>

#include "cpu_memory.h"
#include "dnnl_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "onednn/dnnl.h"

namespace ov {
namespace intel_cpu {

// so far the API only support per-Tensor or per-OC
class DnnlPostOpsComposerLegacy {
public:
    DnnlPostOpsComposerLegacy(const dnnl::engine& engine,
                              dnnl::primitive_attr& attr,
                              dnnl::post_ops& ops,
                              std::unordered_map<int, MemoryPtr>& args,
                              const VectorDims& outputDims,
                              int indexOfOutputChannelDim,
                              bool isINT8,
                              int weiScaleMaskPerChannel,
                              const std::vector<float>& DQScales,
                              bool hasBias);

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

    const VectorDims& getOutputDims() {
        return outputDims;
    }

private:
    const dnnl::engine& engine;
    dnnl::primitive_attr& attr;
    dnnl::post_ops& ops;
    std::unordered_map<int, MemoryPtr>& args;
    const VectorDims outputDims;
    int idxOC;
    const bool isINT8;  // only INT8 primitive support scales
    const int weightScaleMaskPerChannel;
    bool weightScaleAvailable = false;

    VectorDims dimsPerTensor;
    VectorDims dimsPerOC;
    Dim OC;
    int wei_scale_mask = -1;
    std::vector<float> wei_scale_values;
    float dst_scale_val;

    void updateWeiScales();
    void updateDestScales();
};

}  // namespace intel_cpu
}  // namespace ov
