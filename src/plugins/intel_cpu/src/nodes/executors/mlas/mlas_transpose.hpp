// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu_memory.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/transpose.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {
class MlasTransposeExecutor : public TransposeExecutor {
public:
    MlasTransposeExecutor(const TransposeAttrs& attrs, ExecutorContext::CPtr context);
    static bool supports(const TransposeConfig& config);
    static ExecutorPtr create(const TransposeAttrs& attrs,
                              [[maybe_unused]] const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::mlas;
    }

private:
    bool init(const MemoryArgs& memory) override;
    static int64_t calcShapeSize(const Shape& shape, size_t start, size_t end);
    static bool IsTransposeMovingSingleAxis(VectorDims permutations, size_t& from, size_t& to);
    static void TransposeSingleAxisOutwards(const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to);
    static void TransposeSingleAxisInwards(const MemoryCPtr& input, const MemoryPtr& output, size_t from, size_t to);

    size_t from = 0UL;
    size_t to = 0UL;
};

}  // namespace ov::intel_cpu
