// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/paged_selective_ssm_config.hpp"
#include "nodes/executors/selective_ssm_config.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {
class JitKernelBase;
}

namespace ov::intel_cpu {

class SelectiveSSMJitExecutorBase : public Executor {
protected:
    struct ResourceRequirements {
        ov::element::Type data_precision = ov::element::dynamic;
        size_t state_size = 0;
        size_t head_dim_tile = 0;
        size_t state_scratch_elements = 0;
        size_t projection_elements = 0;
        size_t metadata_scratch_elements = 0;
        bool needs_no_state_store_kernel = false;

        [[nodiscard]] size_t projection_scratch_elements() const;
        [[nodiscard]] size_t metadata_scratch_offset() const;
        [[nodiscard]] size_t total_scratch_elements() const;

        bool operator==(const ResourceRequirements& rhs) const {
            return data_precision == rhs.data_precision && state_size == rhs.state_size &&
                   head_dim_tile == rhs.head_dim_tile && state_scratch_elements == rhs.state_scratch_elements &&
                   projection_elements == rhs.projection_elements &&
                   metadata_scratch_elements == rhs.metadata_scratch_elements &&
                   needs_no_state_store_kernel == rhs.needs_no_state_store_kernel;
        }
    };

    struct KernelBundle {
        std::shared_ptr<kernel::JitKernelBase> fp32_state;
        std::shared_ptr<kernel::JitKernelBase> direct_state;
        std::shared_ptr<kernel::JitKernelBase> no_state_store;

        [[nodiscard]] bool ready(bool needs_no_state_store) const {
            return fp32_state != nullptr && direct_state != nullptr &&
                   (!needs_no_state_store || no_state_store != nullptr);
        }
    };

    explicit SelectiveSSMJitExecutorBase(ExecutorContext::CPtr context);

    bool configure_resources(const ResourceRequirements& requirements);
    std::pair<const float*, const float*> prepare_projections(const void* B, const void* C) const;
    [[nodiscard]] impl_desc_type implType() const override;

    ExecutorContext::CPtr m_context;
    KernelBundle m_kernels;
    MemoryPtr m_scratch;
    ResourceRequirements m_requirements;
};

class SelectiveSSMJitExecutor : public SelectiveSSMJitExecutorBase {
public:
    static bool supports(const SelectiveSSMConfig& config);
    static bool accepts_shape(const MemoryArgs& memory);

    SelectiveSSMJitExecutor(const SelectiveSSMAttrs& attrs, const MemoryArgs& memory, ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
};

class PagedSelectiveSSMJitExecutor : public SelectiveSSMJitExecutorBase {
public:
    static bool supports(const PagedSelectiveSSMConfig& config);
    static bool accepts_shape(const MemoryArgs& memory);

    PagedSelectiveSSMJitExecutor(const PagedSelectiveSSMAttrs& attrs,
                                 const MemoryArgs& memory,
                                 ExecutorContext::CPtr context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
};

}  // namespace ov::intel_cpu
