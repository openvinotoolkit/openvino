// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gemm_kernel.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/Strides.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/utils/DataTypeUtils.h>
#include <arm_compute/runtime/NEON/functions/NEGEMM.h>
#include <arm_compute/runtime/Tensor.h>

#include <cstddef>
#include <memory>

#include "nodes/executors/acl/acl_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#define THROW_ERROR(...) OPENVINO_THROW("ACL gemm executor Init Failure '", __VA_ARGS__)

namespace ov::intel_cpu {
GemmKernel::GemmKernel(size_t M, size_t N, size_t K, bool b_transposed, ov::element::Type inType)
    : M(M),
      N(N),
      K(K),
      b_transposed(b_transposed) {
    if (none_of(inType, ov::element::f32, ov::element::f16, ov::element::bf16)) {
        THROW_ERROR("brgemm kernel only supports bf16, f16 and f32");
    }

    if (inType == ov::element::f32) {
        format = arm_compute::Format::F32;
    } else if (inType == ov::element::f16) {
        format = arm_compute::Format::F16;
    } else if (inType == ov::element::bf16) {
        format = arm_compute::Format::BFLOAT16;
    }

    aclGemmKernel = std::make_unique<arm_compute::NEGEMM>();
}

arm_compute::Status GemmKernel::executeGemm(void* a,
                                            void* b,
                                            arm_compute::TensorInfo& dstInfo,
                                            arm_compute::Tensor& dstTensor,
                                            arm_compute::Strides aStrides,
                                            arm_compute::Strides bStrides,
                                            void* c,
                                            float alpha,
                                            float beta,
                                            arm_compute::Strides* outStrides,
                                            void* out) {
    aInfo.init(shapeCast({M, N}),
               format,
               aStrides,
               static_cast<size_t>(0),
               (M * N * arm_compute::element_size_from_data_type(arm_compute::data_type_from_format(format))));

    arm_compute::TensorShape bShape;
    if (b_transposed) {
        bShape = shapeCast({K, N});
    } else {
        bShape = shapeCast({N, K});
    }

    bInfo.init(bShape,
               format,
               bStrides,
               static_cast<size_t>(0),
               (K * N * arm_compute::element_size_from_data_type(arm_compute::data_type_from_format(format))));

    aTensor.allocator()->init(aInfo);
    bTensor.allocator()->init(bInfo);

    if (c != nullptr) {
        cInfo.init(shapeCast({M, K}), format);
        cTensor.allocator()->init(cInfo);
    }

    if (outStrides != nullptr) {
        dstInfo.init(shapeCast({M, K}),
                     format,
                     *outStrides,
                     static_cast<size_t>(0),
                     (M * K * arm_compute::element_size_from_data_type(arm_compute::data_type_from_format(format))));
    } else {
        dstInfo.init(shapeCast({M, K}), format);
    }

    dstTensor.allocator()->init(dstInfo);

    aTensor.allocator()->import_memory(a);
    bTensor.allocator()->import_memory(b);
    cTensor.allocator()->import_memory(c);

    if (out == nullptr) {
        dstTensor.allocator()->allocate();
    } else {
        dstTensor.allocator()->import_memory(out);
    }

    if (b_transposed) {
        aclGemmInfo.set_pretranspose_B(true);
    }

    auto status = aclGemmKernel->validate(&aInfo, &bInfo, &cInfo, &dstInfo, 1.0, 0.0, aclGemmInfo);

    if (c == nullptr) {
        aclGemmKernel->configure(&aTensor, &bTensor, nullptr, &dstTensor, alpha, beta, aclGemmInfo);
    } else {
        aclGemmKernel->configure(&aTensor, &bTensor, &cTensor, &dstTensor, alpha, beta, aclGemmInfo);
    }
    aclGemmKernel->run();

    return status;
}
}  // namespace ov::intel_cpu
