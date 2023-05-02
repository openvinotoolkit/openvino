// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convert.hpp"

namespace ov {
namespace intel_cpu {

class CommonConvertExecutor : public ConvertExecutor {
public:
    explicit CommonConvertExecutor(const ExecutorContext::CPtr context);
    bool init(const ConvertParams& convertParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    impl_desc_type getImplType() const override { return implDescType; };
    ~CommonConvertExecutor() override = default;
protected:
    ConvertParams commonConvertParams;
    impl_desc_type implDescType = impl_desc_type::unknown;
    const ExecutorContext::CPtr convertContext;
};


class CommonConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    ~CommonConvertExecutorBuilder() = default;
    bool isSupported(const ConvertParams& convertParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }
    ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<CommonConvertExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov