// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_convert.hpp"
#include "nodes/common/cpu_convert.h"

ov::intel_cpu::CommonConvertExecutor::CommonConvertExecutor(const ExecutorContext::CPtr context) : ConvertExecutor(context) {}

bool ov::intel_cpu::CommonConvertExecutor::init(const ov::intel_cpu::ConvertParams &convertParams,
                                                const std::vector<MemoryDescPtr> &srcDescs,
                                                const std::vector<MemoryDescPtr> &dstDescs,
                                                const dnnl::primitive_attr &attr) {
    commonConvertParams = convertParams;
    return true;
}

void ov::intel_cpu::CommonConvertExecutor::exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst) {
    cpu_convert(src[0]->GetPtr(),
                dst[0]->GetPtr(),
                commonConvertParams.srcPrc,
                commonConvertParams.origPrc,
                commonConvertParams.dstPrc,
                commonConvertParams.size);
}
