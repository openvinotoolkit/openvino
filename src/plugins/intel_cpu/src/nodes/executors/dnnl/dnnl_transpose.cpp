// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_transpose.hpp"

void ov::intel_cpu::DNNLTransposeExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const int MB) {
    if (!pKernel)
        IE_THROW() << "Could not execute. Kernel for Transpose node was not compiled.";

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(src[0]->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dst[0]->GetPtr());

    pKernel->execute(srcData, dstData, MB);
}

ov::intel_cpu::DNNLTransposeExecutor::DNNLTransposeExecutor(const ExecutorContext::CPtr context) : TransposeExecutor(context) {}

bool ov::intel_cpu::DNNLTransposeExecutor::init(const TransposeParams &transposeParams,
                                                const std::vector<MemoryDescPtr> &srcDescs,
                                                const std::vector<MemoryDescPtr> &dstDescs, const dnnl::primitive_attr &attr) {
    if (transposeParams.transposeExecution != TransposeParams::NOT_REF) { return false; }
    pKernel = std::make_shared<PermuteKernel>(transposeParams.permuteParams);
    return true;
}