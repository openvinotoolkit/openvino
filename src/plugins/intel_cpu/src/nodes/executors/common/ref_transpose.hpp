// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/transpose.hpp"

namespace ov::intel_cpu {
class RefTransposeExecutor : public TransposeExecutor {
public:
    using TransposeExecutor::TransposeExecutor;
    static void referenceExecute(const uint8_t* src_data,
                                 uint8_t* dst_data,
                                 const jit_permute_config_params& jcp,
                                 const int mb);
    bool init(const TransposeParams& transposeParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

private:
    jit_permute_config_params jcp;
};

class RefTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    [[nodiscard]] bool isSupported(const TransposeParams& transposeParams,
                                   const std::vector<MemoryDescPtr>& srcDescs,
                                   const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    [[nodiscard]] TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefTransposeExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
