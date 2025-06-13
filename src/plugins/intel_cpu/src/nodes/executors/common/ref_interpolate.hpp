// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/executor.hpp"
#include "onednn/iml_type_mapper.h"
#include "ref_interpolate_legacy.hpp"

namespace ov::intel_cpu {

class CommonInterpolateExecutor : public Executor {
public:
    CommonInterpolateExecutor(const InterpolateAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr context);

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const InterpolateConfig& config);

private:
    std::vector<const void*> postOpsDataPtrs_;
    std::shared_ptr<legacy::InterpolateRefExecutorLegacy> refExecutorLegacy;
};
using CommonInterpolateExecutorPtr = std::shared_ptr<CommonInterpolateExecutor>;

}  // namespace ov::intel_cpu
