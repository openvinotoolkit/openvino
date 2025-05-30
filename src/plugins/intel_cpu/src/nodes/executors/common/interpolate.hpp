// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "cpu_types.h"
#include "nodes/executors/interpolate_config.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

class InterpolateRefExecutor : public InterpolateExecutorBase {
public:
    InterpolateRefExecutor(const InterpolateAttrs& interpAttrs,
                           const VectorDims& srcDims,
                           const VectorDims& dstDims,
                           const std::vector<float>& _dataScales)
        : InterpolateExecutorBase(interpAttrs, srcDims, dstDims, _dataScales),
          antialias(interpAttrs.antialias),
          dataScales(_dataScales),
          refInterpAttrs(interpAttrs) {}

    void exec(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_) override;

private:
    void NNRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int ID, int IH, int IW, int OD, int OH, int OW);

    void linearOnnxRef(const uint8_t* in_ptr_,
                       uint8_t* out_ptr_,
                       int B,
                       int C,
                       int ID,
                       int IH,
                       int IW,
                       int OD,
                       int OH,
                       int OW);

    void cubicRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

    void linearInterpolation(const uint8_t* in_ptr_,
                             uint8_t* out_ptr_,
                             int B,
                             int C,
                             int ID,
                             int IH,
                             int IW,
                             float fx,
                             float fy,
                             float fz,
                             int OD,
                             int OH,
                             int OW,
                             int kernel_width,
                             bool antialias);

    void pillowRef(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

    void pillowRefNCHWAsNHWC(const uint8_t* in_ptr_, uint8_t* out_ptr_, int B, int C, int IH, int IW, int OH, int OW);

    static float getValue(const uint8_t* base, size_t offset, ov::element::Type prec);

    static void setValue(uint8_t* base, size_t offset, float value, ov::element::Type prec);

private:
    bool antialias;
    std::vector<float> dataScales;
    InterpolateAttrs refInterpAttrs;
};

}  // namespace ov::intel_cpu
