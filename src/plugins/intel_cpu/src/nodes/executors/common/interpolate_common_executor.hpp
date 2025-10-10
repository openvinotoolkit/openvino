// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/interpolate_config.hpp"

namespace ov::intel_cpu {

class InterpolateCommonExecutor : public Executor {
public:
    InterpolateCommonExecutor(InterpolateAttrs attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override { return impl_desc_type::ref; }

private:
    InterpolateAttrs m_attrs;
    ExecutorContext::CPtr m_context;
    std::shared_ptr<InterpolateExecutorBase> m_ref;
    std::vector<int> m_conversion5DMap;
};

}  // namespace ov::intel_cpu
