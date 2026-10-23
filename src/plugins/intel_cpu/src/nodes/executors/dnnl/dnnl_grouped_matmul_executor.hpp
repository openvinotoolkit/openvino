// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "nodes/executors/dnnl/dnnl_inner_product_gemm.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/grouped_matmul_config.hpp"
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
    InnerProductPtr impl_for(Dim rows);

    ExecutorContext::CPtr m_context;

    MemoryPtr m_weights_memory;
    MemoryPtr m_scales_memory;
    MemoryPtr m_zp_memory;

    dnnl_utils::InnerProductKey m_key_template;
    dnnl::memory::data_type m_src_data_type = dnnl::memory::data_type::undef;
    dnnl::memory::dim m_K = 0;

    // true for the 3D x 3D form: mat_a is [G, M, K] and every group owns exactly M rows.
    // false for the 2D x 3D form: mat_a is [T, K] and the row ranges come from the offsets input.
    bool m_is_batched = false;
    impl_desc_type m_impl_type = impl_desc_type::unknown;
};

}  // namespace ov::intel_cpu
