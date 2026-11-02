// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/qsa_indexer_config.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

// The portable implementation: plain C++, the scalar scoring of the blocks.
class QSAIndexerRefExecutor : public Executor {
public:
    static bool supports(const QSAIndexerConfig& config);

    QSAIndexerRefExecutor(const QSAIndexerAttrs& attrs, const MemoryArgs& /*memory*/, ExecutorContext::CPtr context)
        : m_attrs(attrs),
          m_context(std::move(context)) {}

    bool update(const MemoryArgs& /*memory*/) override {
        return true;
    }
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

private:
    QSAIndexerAttrs m_attrs;
    ExecutorContext::CPtr m_context;
};

}  // namespace ov::intel_cpu
