// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_matmul.hpp"
#include "acl_utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclMatMulExecutor::AclMatMulExecutor(const ExecutorContext::CPtr context) : MatMulExecutor(context) {}

bool AclMatMulExecutor::init(const MatMulAttrs& matmulAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto weiDims = srcDescs[1]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    auto srcBatch = vectorProduct(srcDims, srcDims.size() - 2);
    auto weiBatch = vectorProduct(weiDims, weiDims.size() - 2);
    auto dstBatch = vectorProduct(dstDims, dstDims.size() - 2);
    auto M = srcDims[srcDims.size() - 2];
    auto K = srcDims[srcDims.size() - 1];
    auto N = weiDims[weiDims.size() - 1];

    // ACL doesn't support cases then both inputs are broadcasted
    if (srcBatch > 1 && weiBatch > 1 && srcBatch != weiBatch ||
        srcBatch != dstBatch && weiBatch != dstBatch) {
        return false;
    }

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(K, M, 1, srcBatch), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo weiTensorInfo = TensorInfo(TensorShape(N, K, weiBatch), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(N, M, 1, dstBatch), 1, DataType::F32, DataLayout::NCHW);

    if (!arm_compute::NEGEMM::validate(&srcTensorInfo, &weiTensorInfo, nullptr, &dstTensorInfo, 1.0f, 0.0f))
        return false;

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    matmul = std::make_unique<arm_compute::NEGEMM>();
    matmul->configure(&srcTensor, &weiTensor, nullptr, &dstTensor, 1.0f, 0.0f);

    return true;
}

void AclMatMulExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    weiTensor.allocator()->import_memory(src[1]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());

    matmul->run();

    srcTensor.allocator()->free();
    weiTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
