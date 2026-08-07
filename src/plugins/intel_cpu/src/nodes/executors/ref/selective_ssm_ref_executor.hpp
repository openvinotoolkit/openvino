// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/executor.hpp"
#include "nodes/executors/selective_ssm_config.hpp"

namespace ov::intel_cpu {

class SelectiveSSMRefExecutor : public Executor {
public:
    static bool supports(const SelectiveSSMConfig& config);

    SelectiveSSMRefExecutor(const SelectiveSSMAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    SelectiveSSMAttrs m_attrs;
};

}  // namespace ov::intel_cpu
