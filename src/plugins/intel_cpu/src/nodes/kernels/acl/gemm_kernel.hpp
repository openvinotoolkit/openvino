// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <cstddef>
#include <openvino/core/type/element_type.hpp>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "nodes/executors/acl/acl_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
class GemmKernel {
public:
    GemmKernel(size_t M, size_t N, size_t K, bool b_transposed = false, ov::element::Type inType = ov::element::f32);

    arm_compute::Status executeGemm(void* a,
                                    void* b,
                                    arm_compute::TensorInfo& dstInfo,
                                    arm_compute::Tensor& dstTensor,
                                    arm_compute::Strides aStrides,
                                    arm_compute::Strides bStrides,
                                    void* c = nullptr,
                                    float alpha = 1.0f,
                                    float beta = 0.0f,
                                    arm_compute::Strides* outStrides = nullptr,
                                    void* out = nullptr);

private:
    size_t M = 0;
    size_t N = 0, K = 0;
    bool b_transposed = false;
    arm_compute::Format format;
    arm_compute::TensorInfo aInfo;
    arm_compute::TensorInfo bInfo;
    arm_compute::TensorInfo cInfo;
    arm_compute::Tensor aTensor;
    arm_compute::Tensor bTensor;
    arm_compute::Tensor cTensor;
    arm_compute::Tensor dTensor;
    std::unique_ptr<arm_compute::NEGEMM> aclGemmKernel;
    arm_compute::GEMMInfo aclGemmInfo;
};

}  // namespace ov::intel_cpu
