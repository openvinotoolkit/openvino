// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// @file dnnl_utils.hpp
// Contains utility methods used by oneDNN backend executors
//

#pragma once

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"
#include "nodes/executors/executor.hpp"
#include "dnnl_fullyconnected_primitive.hpp"

namespace ov {
namespace intel_cpu {
namespace utils {

template <typename Primitive>
DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr srcDesc,
                                                 const DnnlMemoryDescPtr dstDesc,
                                                 bool weightsNonTransposed) {
    if (!weightsNonTransposed)
        return srcDesc;

    const auto& weiDesc = srcDesc->getDnnlDesc();
    const auto reorderedWeiDesc = dnnl::memory::desc{weiDesc.get_dims(), weiDesc.get_data_type(), dnnl::memory::format_tag::ba};
    const auto transposedWeiDesc = reorderedWeiDesc.reshape(dstDesc->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(transposedWeiDesc);
}

template <>
DnnlMemoryDescPtr makeTransposedWeightDescriptor<DnnlFCPrimitive>(const DnnlMemoryDescPtr srcDesc,
                                                                  const DnnlMemoryDescPtr dstDesc,
                                                                  bool weightsNonTransposed);

template <>
DnnlMemoryDescPtr makeTransposedWeightDescriptor<DnnlMatMulPrimitive>(const DnnlMemoryDescPtr srcDesc,
                                                                      const DnnlMemoryDescPtr dstDesc,
                                                                      bool weightsNonTransposed);

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr srcWeightDesc,
                               const DnnlMemoryDescPtr dstWeightDesc,
                               const MemoryCPtr weightsMem,
                               const ExecutorContext::CPtr context,
                               const bool needShiftSignedToUnsigned = false);
}  // namespace utils
}  // namespace intel_cpu
}  // namespace ov
