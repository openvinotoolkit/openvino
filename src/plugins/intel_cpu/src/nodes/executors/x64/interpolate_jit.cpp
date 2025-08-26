// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Modified to support dynamic shapes

#include "interpolate_jit.hpp"
#include "nodes/interpolate.h"
#include "cpu_shape.h"
#include "utils/general_utils.h"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>
#include <dnnl_extension_utils.h>
#include <cmath>
#include <algorithm>

namespace ov::intel_cpu {

using namespace dnnl::impl::cpu;
using namespace node;

JitInterpolateExecutor::JitInterpolateExecutor(ExecutorContext::CPtr context) 
    : InterpolateExecutor(context) {
}

bool JitInterpolateExecutor::init(const InterpolateAttrs& interpolateAttrs,
                                  const std::vector<MemoryDescPtr>& srcDescs,
                                  const std::vector<MemoryDescPtr>& dstDescs,
                                  const dnnl::primitive_attr& attr) {
    // Call base class init first
    if (!InterpolateExecutor::init(interpolateAttrs, srcDescs, dstDescs, attr)) {
        return false;
    }
    
    attrs_ = interpolateAttrs;
    
    // Extract dimensions and precision
    if (!srcDescs.empty()) {
        srcDims_ = srcDescs[0]->getShape().getStaticDims();
        dataPrecision_ = srcDescs[0]->getPrecision();
    }
    if (!dstDescs.empty()) {
        dstDims_ = dstDescs[0]->getShape().getStaticDims();
    }
    
    // Check if we have padding
    hasPadding_ = false;
    for (size_t i = 0; i < attrs_.padBegin.size(); i++) {
        if (attrs_.padBegin[i] != 0 || attrs_.padEnd[i] != 0) {
            hasPadding_ = true;
            break;
        }
    }
    
    // Calculate padded dimensions if needed
    if (hasPadding_) {
        srcDimsPadded_ = getPaddedInputShape(srcDims_, attrs_.padBegin, attrs_.padEnd);
    } else {
        srcDimsPadded_ = srcDims_;
    }
    
    // Calculate or use provided scales
    dataScales_ = attrs_.dataScales;
    if (dataScales_.empty() && !srcDimsPadded_.empty() && !dstDims_.empty()) {
        dataScales_.reserve(srcDimsPadded_.size());
        for (size_t i = 0; i < srcDimsPadded_.size(); i++) {
            dataScales_.push_back(static_cast<float>(dstDims_[i]) / static_cast<float>(srcDimsPadded_[i]));
        }
    }
    
    // Create the old JIT executor to leverage existing implementation
    try {
        oldJitExecutor_ = std::make_shared<Interpolate::OldInterpolateJitExecutor>(
            attrs_,
            srcDimsPadded_,
            dstDims_,
            dataScales_,
            attr);
        return true;
    } catch (const std::exception& e) {
        // Failed to create JIT executor
        oldJitExecutor_ = nullptr;
        return false;
    }
}


bool JitInterpolateExecutor::update(const MemoryArgs& memory) {
    // When shapes change, we need to recreate the JIT executor with new dimensions
    if (!memory.count(ARG_SRC_0) || !memory.count(ARG_DST)) {
        return false;
    }
    
    // Get new dimensions from memory
    auto srcMem = memory.at(ARG_SRC_0);
    auto dstMem = memory.at(ARG_DST);
    
    VectorDims newSrcDims = srcMem->getStaticDims();
    VectorDims newDstDims = dstMem->getStaticDims();
    
    // Check if dimensions changed
    bool dimsChanged = (newSrcDims != srcDims_ || newDstDims != dstDims_);
    
    if (!dimsChanged && oldJitExecutor_) {
        // No change needed, keep existing executor
        return true;
    }
    
    // Update dimensions
    srcDims_ = newSrcDims;
    dstDims_ = newDstDims;
    
    // Recalculate padded dimensions if needed
    if (hasPadding_) {
        srcDimsPadded_ = getPaddedInputShape(srcDims_, attrs_.padBegin, attrs_.padEnd);
    } else {
        srcDimsPadded_ = srcDims_;
    }
    
    // Recalculate scales
    dataScales_.clear();
    if (!srcDimsPadded_.empty() && !dstDims_.empty()) {
        dataScales_.reserve(srcDimsPadded_.size());
        for (size_t i = 0; i < srcDimsPadded_.size(); i++) {
            dataScales_.push_back(static_cast<float>(dstDims_[i]) / static_cast<float>(srcDimsPadded_[i]));
        }
    }
    
    // Create new JIT executor with updated dimensions
    try {
        // Get the primitive attribute from memory descriptors
        dnnl::primitive_attr attr;
        
        oldJitExecutor_ = std::make_shared<Interpolate::OldInterpolateJitExecutor>(
            attrs_,
            srcDimsPadded_,
            dstDims_,
            dataScales_,
            attr);
        return true;
    } catch (const std::exception& e) {
        oldJitExecutor_ = nullptr;
        return false;
    }
}

impl_desc_type JitInterpolateExecutor::getImplType() const {
#if defined(OPENVINO_ARCH_X86_64)
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        return impl_desc_type::jit_avx512;
    }
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return impl_desc_type::jit_avx2;
    }
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        return impl_desc_type::jit_sse42;
    }
#endif
    return impl_desc_type::jit;
}

void JitInterpolateExecutor::exec(const std::vector<MemoryCPtr>& src,
                                  const std::vector<MemoryPtr>& dst,
                                  const void* post_ops_data_) {
    if (!oldJitExecutor_ || src.empty() || dst.empty()) {
        return;
    }
    
    // Use padPreprocess from base class to handle padding
    const uint8_t* src_data = padPreprocess(src, dst);
    uint8_t* dst_data = dst[0]->getDataAs<uint8_t>();
    
    // Execute using the old JIT executor
    oldJitExecutor_->exec(src_data, dst_data, post_ops_data_);
}


}  // namespace ov::intel_cpu