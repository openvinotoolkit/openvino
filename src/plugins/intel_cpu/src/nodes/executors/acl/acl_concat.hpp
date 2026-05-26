// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/runtime/NEON/functions/NEConcatenateLayer.h>
#include <arm_compute/runtime/Tensor.h>

#include "nodes/executors/concat_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov::intel_cpu {

class AclConcatExecutor : public Executor {
public:
    AclConcatExecutor(const ConcatAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    static bool supports(const ConcatConfig& config, LayoutType expectedLayout);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::acl;
    }

private:
    ConcatAttrs m_attrs;
    LayoutType m_expectedLayout;
    std::vector<int> m_srcArgIds;
    std::vector<arm_compute::Tensor> m_srcTensors;
    arm_compute::Tensor m_dstTensor;
    arm_compute::NEConcatenateLayer m_concatLayer;
};

}  // namespace ov::intel_cpu
