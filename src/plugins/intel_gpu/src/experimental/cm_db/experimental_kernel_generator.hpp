// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Experimental CM kernel generator base.
//
// Upstream's `ov::intel_gpu::cm::KernelGenerator` hard-codes template lookup through the upstream
// `SourcesDB` (impls/cm/utils/kernels_db.{hpp,cpp}), whose strings come from the upstream
// `gpu_cm_kernel_sources.inc`. To port the BlockSparseAttention / DraftBlockMask CM kernels without
// registering any new kernel in that upstream DB, this base replicates the upstream
// `KernelGenerator::get_kernel_data()` flow but resolves templates through
// `experimental::cm::ExperimentalSourcesDB` instead.
//
// The experimental impl generator classes derive from `ExperimentalKernelGenerator` rather than from
// the upstream `KernelGenerator` (only the base class needs to change -- every virtual hook they
// override, e.g. get_jit_constants()/get_arguments_desc()/get_dispatch_data_func(), is preserved).

#pragma once

#include <string_view>

#include "impls/cm/utils/kernel_generator.hpp"

namespace ov::intel_gpu::experimental::cm {

class ExperimentalKernelGenerator : public ov::intel_gpu::cm::KernelGenerator {
public:
    explicit ExperimentalKernelGenerator(std::string_view name, std::string_view suffix = "")
        : KernelGenerator(name, suffix), m_kernel_name(name) {}

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override;

private:
    std::string m_kernel_name;
};

}  // namespace ov::intel_gpu::experimental::cm