// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/convert.hpp"
#include "nodes/executors/executor.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

class CommonConvertExecutor : public ConvertExecutor {
public:
    using ConvertExecutor::ConvertExecutor;
    bool init(const ConvertParams& convertParams,
              const MemoryDescPtr& srcDesc,
              const MemoryDescPtr& dstDesc,
              const dnnl::primitive_attr& attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return implDescType;
    };
    static bool isSupported(ov::element::Type srcPrc, ov::element::Type dstPrc);

protected:
    ConvertParams commonConvertParams;
    static const impl_desc_type implDescType = impl_desc_type::ref;
    const ExecutorContext::CPtr convertContext;
};

class CommonConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    ~CommonConvertExecutorBuilder() override = default;
    [[nodiscard]] bool isSupported([[maybe_unused]] const ConvertParams& convertParams,
                                   [[maybe_unused]] const MemoryDescPtr& srcDesc,
                                   [[maybe_unused]] const MemoryDescPtr& dstDesc) const override {
        return true;
    }
    [[nodiscard]] ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<CommonConvertExecutor>(context);
    }
};

}  // namespace ov::intel_cpu
