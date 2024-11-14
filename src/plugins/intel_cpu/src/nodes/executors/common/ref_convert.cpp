// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_convert.hpp"
#include "nodes/common/cpu_convert.h"

namespace ov {
namespace intel_cpu {

bool CommonConvertExecutor::isSupported(ov::element::Type srcPrc, ov::element::Type dstPrc) {
    return is_supported_convert(srcPrc, dstPrc);
}

bool CommonConvertExecutor::init(const ConvertParams& convertParams,
                                                const MemoryDescPtr& srcDesc,
                                                const MemoryDescPtr& dstDesc,
                                                const dnnl::primitive_attr& attr) {
    commonConvertParams = convertParams;
    return true;
}

void CommonConvertExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    assert(src.size() == 1);
    assert(dst.size() == 1);

    cpu_convert(src[0]->getData(),
                dst[0]->getData(),
                commonConvertParams.srcPrc,
                commonConvertParams.origPrc,
                commonConvertParams.dstPrc,
                commonConvertParams.size);
}

} // namespace intel_cpu
} // namespace ov