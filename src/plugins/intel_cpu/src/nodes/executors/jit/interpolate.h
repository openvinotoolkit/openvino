// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <common/primitive_attr.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "nodes/executors/interpolate_config.hpp"

namespace ov::intel_cpu {

struct jit_interpolate_config_params {
    InterpolateLayoutType layout = InterpolateLayoutType::planar;
    InterpolateMode mode = InterpolateMode::nearest;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    int src_data_size = 0;
    int dst_data_size = 0;
    int indices_size = 0;
    int spatial_dim_size = 0;
    int C = 0, ID = 0, IH = 0, IW = 0, OD = 0, OH = 0, OW = 0;
    // for pillow
    int filterLenX = 0;
    int filterLenY = 0;
    int* bound = nullptr;
};

struct jit_interpolate_call_args {
    const void* src_ptr[MAX_INPUT_INTERPOLATE];
    const void* weight_ptr[MAX_INPUT_INTERPOLATE];
    const int* index;
    void* dst;
    size_t work_amount;
    size_t oc_off;
    // ptr to array of post op inputs pointers (flat list)
    const void* post_op_data;
};

struct jit_uni_interpolate_kernel {
    void (*ker_)(const jit_interpolate_call_args*) = nullptr;

    void operator()(const jit_interpolate_call_args* args) const {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_interpolate_kernel(jit_interpolate_config_params jcp, const dnnl_primitive_attr& attr)
        : jcp_(jcp),
          attr_(attr) {}
    virtual ~jit_uni_interpolate_kernel() = default;

    virtual void create_ker() = 0;

    jit_interpolate_config_params jcp_;
    const dnnl_primitive_attr& attr_;
};

class InterpolateJitExecutor : public InterpolateExecutorBase {
public:
    InterpolateJitExecutor(const InterpolateAttrs& interpAttrs,
                           const VectorDims& srcDims,
                           const VectorDims& dstDims,
                           const std::vector<float>& dataScales,
                           const dnnl::primitive_attr& attr);

    void exec(const uint8_t* in_ptr_, uint8_t* out_ptr_, const void* post_ops_data_) override;

private:
    // nearest neighbor
    void NNPlanar(const uint8_t* in_ptr_,
                  uint8_t* out_ptr_,
                  const void* post_ops_data_,
                  int B,
                  int C,
                  int ID,
                  int IH,
                  int IW,
                  int OD,
                  int OH,
                  int OW);
    void NNCGathered(const uint8_t* in_ptr_,
                     uint8_t* out_ptr_,
                     const void* post_ops_data_,
                     int B,
                     int C,
                     int ID,
                     int IH,
                     int IW,
                     int OD,
                     int OH,
                     int OW);

    // onnx linear
    void linearOnnxPlanar(const uint8_t* in_ptr_,
                          uint8_t* out_ptr_,
                          const void* post_ops_data_,
                          int B,
                          int C,
                          int ID,
                          int IH,
                          int IW,
                          int OD,
                          int OH,
                          int OW);
    void linearOnnxCGathered(const uint8_t* in_ptr_,
                             uint8_t* out_ptr_,
                             const void* post_ops_data_,
                             int B,
                             int C,
                             int ID,
                             int IH,
                             int IW,
                             int OD,
                             int OH,
                             int OW);

    // cubic
    void cubicPlanar(const uint8_t* in_ptr_,
                     uint8_t* out_ptr_,
                     const void* post_ops_data_,
                     int B,
                     int C,
                     int IH,
                     int IW,
                     int OH,
                     int OW);
    void cubicCGathered(const uint8_t* in_ptr_,
                        uint8_t* out_ptr_,
                        const void* post_ops_data_,
                        int B,
                        int C,
                        int IH,
                        int IW,
                        int OH,
                        int OW);

    // pillow bilinear and pillow bicubic
    void pillowCGathered(const uint8_t* in_ptr_,
                         uint8_t* out_ptr_,
                         const void* post_ops_data_,
                         int B,
                         int C,
                         int IH,
                         int IW,
                         int OH,
                         int OW);

    std::shared_ptr<jit_uni_interpolate_kernel> interpolateKernel = nullptr;
};

}