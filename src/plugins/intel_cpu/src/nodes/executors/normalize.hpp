// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include <dnnl_extension_utils.h>

namespace ov {
namespace intel_cpu {

enum class NormEpsMode {
    ADD,
    MAX
};

struct NormalizeL2Attrs {
    LayoutType layout = LayoutType::ncsp;
    NormEpsMode epsMode = NormEpsMode::ADD;
    bool across_spatial = true;
    bool cornerCase = false;
    float eps = 1e-10f;

    InferenceEngine::Precision input_prec = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision output_prec = InferenceEngine::Precision::UNSPECIFIED;
    size_t src_data_size = 0lu;
    size_t dst_data_size = 0lu;
    VectorDims vectorDims;
    impl_desc_type implDescType;
    AxisSet axisSet;
    bool isFusing = false;
};

struct NormalizeKey {
    NormalizeL2Attrs attrs;
    dnnl::primitive_attr kernel_attrs;
    size_t hash() const;
    bool operator==(const NormalizeKey& rhs) const;
};

class NormalizeL2Executor {
public:
    explicit NormalizeL2Executor(const ExecutorContext::CPtr context);
    virtual bool init(const NormalizeL2Attrs& normalizeL2Attrs,
                      const std::vector<MemoryDescPtr>& srcDescs,
                      const std::vector<MemoryDescPtr>& dstDescs,
                      const dnnl::primitive_attr &attr) = 0;
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void **post_ops_data_) = 0;
    virtual ~NormalizeL2Executor() = default;
    virtual impl_desc_type getImplType() const = 0;
protected:
    NormalizeL2Attrs normalizeL2Attrs;
    const ExecutorContext::CPtr implContext;
    static inline float epsApply(const float &modulo, const NormEpsMode mode, const float eps) {
        return mode == NormEpsMode::ADD ? modulo + eps : std::max(modulo, eps);
    }
};
using NormalizeL2ExecutorPtr = std::shared_ptr<NormalizeL2Executor>;
using NormalizeL2ExecutorCPtr = std::shared_ptr<const NormalizeL2Executor>;

class NormalizeL2ExecutorBuilder {
public:
    ~NormalizeL2ExecutorBuilder() = default;
    virtual bool isSupported(const NormalizeL2Attrs& normalizeL2Attrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    virtual NormalizeL2ExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const = 0;
};

using NormalizeL2ExecutorBuilderPtr = std::shared_ptr<NormalizeL2ExecutorBuilder>;
using NormalizeL2ExecutorBuilderCPtr = std::shared_ptr<const NormalizeL2ExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov