// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"

namespace ov {
namespace intel_cpu {
class RefTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;
    static void referenceExecute(const uint8_t* src_data, uint8_t* dst_data, jit_permute_config_params jcp, const int mb);
    bool init(const TransposeParams &transposeParams,
              const std::vector<MemoryDescPtr> &srcDescs,
              const std::vector<MemoryDescPtr> &dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return implType; }
private:
    static const impl_desc_type implType = impl_desc_type::ref;
    jit_permute_config_params jcp;
};

class RefTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const TransposeParams& transposeParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefTransposeExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov
