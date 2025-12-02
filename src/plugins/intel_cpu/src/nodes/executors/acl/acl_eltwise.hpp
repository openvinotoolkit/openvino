// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include <memory>
#include <vector>

#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class AclEltwiseExecutor : public Executor {
public:
    AclEltwiseExecutor(EltwiseAttrs attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);
    static bool supports(const EltwiseConfig& config);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return m_implType;
    }

private:
    bool init(const std::vector<MemoryDescPtr>& srcDescs, const std::vector<MemoryDescPtr>& dstDescs);

    EltwiseAttrs aclEltwiseAttrs;
    impl_desc_type m_implType = impl_desc_type::acl;
    std::vector<arm_compute::Tensor> srcTensors, dstTensors;
    std::unique_ptr<arm_compute::IFunction> ifunc;
};

}  // namespace ov::intel_cpu
