// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "nodes/executors/transpose.hpp"

namespace ov::intel_cpu {

class ReorderTransposeExecutor : public TransposeExecutor {
public:
    ReorderTransposeExecutor(const TransposeAttrs& attrs, ExecutorContext::CPtr context);

    static bool supports(const TransposeConfig& config);
    static ExecutorPtr create(const TransposeAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);

    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return m_implType;
    }

private:
    bool init(const MemoryArgs& memory) override;

    dnnl::reorder m_primitive;
    impl_desc_type m_implType = impl_desc_type::undef;
};

}  // namespace ov::intel_cpu
