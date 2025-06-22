// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "executor.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"
#include "mvn_config.hpp"

namespace ov::intel_cpu {

class MVNExecutorBuilder {
public:
    virtual ~MVNExecutorBuilder() = default;
    [[nodiscard]] virtual bool isSupported(const MVNAttrs& mvnAttrs,
                                           const std::vector<MemoryDescPtr>& srcDescs,
                                           const std::vector<MemoryDescPtr>& dstDescs) const = 0;
    [[nodiscard]] virtual legacy::MVNExecutorPtr makeExecutor(ExecutorContext::CPtr context) const = 0;
};

using MVNExecutorBuilderPtr = std::shared_ptr<MVNExecutorBuilder>;
using MVNExecutorBuilderCPtr = std::shared_ptr<const MVNExecutorBuilder>;

}  // namespace ov::intel_cpu
