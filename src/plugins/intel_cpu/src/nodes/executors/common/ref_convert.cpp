// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ref_convert.hpp"

#include <cassert>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/executors/convert.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

bool CommonConvertExecutor::isSupported(ov::element::Type srcPrc, ov::element::Type dstPrc) {
    return is_supported_convert(srcPrc, dstPrc);
}

bool CommonConvertExecutor::init(const ConvertParams& convertParams,
                                 [[maybe_unused]] const MemoryDescPtr& srcDesc,
                                 [[maybe_unused]] const MemoryDescPtr& dstDesc,
                                 [[maybe_unused]] const dnnl::primitive_attr& attr) {
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
                commonConvertParams.size,
                commonConvertParams.bypass_clamp);
}

}  // namespace ov::intel_cpu
