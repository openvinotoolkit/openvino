// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convert.hpp"

namespace ov {
namespace intel_cpu {

class CommonConvertExecutor : public ConvertExecutor {
public:
    using ConvertExecutor::ConvertExecutor;
    bool init(const ConvertParams& convertParams,
              const MemoryDescPtr& srcDesc,
              const MemoryDescPtr& dstDesc,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    impl_desc_type implType() const override { return implDescType; };
    static bool isSupported(ov::element::Type srcPrc, ov::element::Type dstPrc);

protected:
    ConvertParams commonConvertParams;
    static const impl_desc_type implDescType = impl_desc_type::ref;
    const ExecutorContext::CPtr convertContext;
};


class CommonConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    ~CommonConvertExecutorBuilder() = default;
    bool isSupported(const ConvertParams& convertParams,
                     const MemoryDescPtr& srcDesc,
                     const MemoryDescPtr& dstDesc) const override {
        return true;
    }
    ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<CommonConvertExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov
