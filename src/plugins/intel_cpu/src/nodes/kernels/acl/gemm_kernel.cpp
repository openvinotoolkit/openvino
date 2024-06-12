// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "gemm_kernel.hpp"
#define THROW_ERROR(...) OPENVINO_THROW("ACL gemm executor Init Failure '", __VA_ARGS__)

using Dim = std::size_t;
using VectorDims = std::vector<Dim>;
inline arm_compute::TensorShape shapeCast(const VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], true);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

namespace ov {
namespace intel_cpu {
    GemmKernel::GemmKernel(size_t M,
                           size_t N,
                           size_t K,
                           bool b_transposed,
                           ov::element::Type inType)
    : M(M),
      N(N),
      K(K),
      b_transposed(b_transposed) {
        if (inType != ov::element::f32 && inType != ov::element::f16 && inType != ov::element::bf16)
            THROW_ERROR("brgemm kernel only supports bf16, f16 and f32");

        if (inType == ov::element::f32)
            format = arm_compute::Format::F32;
        else if (inType == ov::element::f16)
            format = arm_compute::Format::F16;
        else if (inType == ov::element::bf16)
            format = arm_compute::Format::BFLOAT16;


        aclGemmKernel = std::make_unique<arm_compute::NEGEMM>();
    }

    arm_compute::Status GemmKernel::executeGemm(void *a,
                                                void *b,
                                                arm_compute::TensorInfo& dstInfo,
                                                arm_compute::Tensor& dstTensor,
                                                void *c,
                                                float alpha,
                                                float beta,
                                                arm_compute::Strides* outStrides,
                                                void* out) {
        aInfo.init(shapeCast({M, N}), format);
        if (b_transposed)
            bInfo.init(shapeCast({K, N}), format);
        else
            bInfo.init(shapeCast({N, K}), format);

        aTensor.allocator()->init(aInfo);
        bTensor.allocator()->init(bInfo);

        if (c != nullptr) {
            cInfo.init(shapeCast({M, K}), format);
            cTensor.allocator()->init(cInfo);
        }

        if (outStrides != nullptr)
            dstInfo.init(
                    shapeCast({M, K}),
                    format,
                    *outStrides,
                    size_t(0),
                    (size_t)(M * K * arm_compute::element_size_from_data_type(arm_compute::data_type_from_format(format))));
        else
            dstInfo.init(shapeCast({M, K}), format);

        dstTensor.allocator()->init(dstInfo);

        aTensor.allocator()->import_memory(reinterpret_cast<void *>(a));
        bTensor.allocator()->import_memory(reinterpret_cast<void *>(b));
        cTensor.allocator()->import_memory(reinterpret_cast<void *>(c));

        if (out == nullptr)
            dstTensor.allocator()->allocate();
        else
            dstTensor.allocator()->import_memory(out);

        if (b_transposed)
            aclGemmInfo.set_pretranspose_B(true);

        auto status = aclGemmKernel->validate(&aInfo, &bInfo, &cInfo, &dstInfo, 1.0, 0.0, aclGemmInfo);

        if (c == nullptr)
            aclGemmKernel->configure(&aTensor, &bTensor, nullptr, &dstTensor, alpha, beta, aclGemmInfo);
        else
            aclGemmKernel->configure(&aTensor, &bTensor, &cTensor, &dstTensor, alpha, beta, aclGemmInfo);
        aclGemmKernel->run();

        return status;
    }
} // namespace intel_cpu
} // namespace ov