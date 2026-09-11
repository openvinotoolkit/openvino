// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "nodes/executors/dnnl/dnnl_inner_product_gemm.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/groupedmatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

class GroupedMatMulDnnlExecutor : public Executor {
public:
    static bool supports(const GroupedMatMulConfig& config);

    GroupedMatMulDnnlExecutor(const GroupedMatMulAttrs& attrs,
                              const MemoryArgs& memory,
                              const ExecutorContext::CPtr& context);

    bool update(const MemoryArgs& memory) override;
    void execute(const MemoryArgs& memory) override;
    [[nodiscard]] impl_desc_type implType() const override;

private:
    using InnerProductPtr = dnnl_utils::InnerProductPtr;

    // The inner_product for a group of `rows` rows, taken from the runtime cache and created on
    // first use. See the comment on the definition for why this happens during execute().
    InnerProductPtr implFor(Dim rows);

    ExecutorContext::CPtr m_context;

    MemoryPtr m_weightsMemory;
    MemoryPtr m_scalesMemory;
    MemoryPtr m_zpMemory;

    dnnl_utils::InnerProductKey m_keyTemplate;
    dnnl::memory::data_type m_srcDataType = dnnl::memory::data_type::undef;
    dnnl::memory::dim m_K = 0;

    // true for the 3D x 3D form: mat_a is [G, M, K] and every group owns exactly M rows.
    // false for the 2D x 3D form: mat_a is [T, K] and the row ranges come from the offsets input.
    bool m_isBatched = false;
    impl_desc_type m_implType = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
