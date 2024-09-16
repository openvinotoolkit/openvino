// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"

namespace ov::intel_gpu::cm {

class KernelGenerator : public KernelGeneratorBase {
public:
    explicit KernelGenerator(std::string_view name) : KernelGeneratorBase(), m_kernel_name(name) {}
    virtual ~KernelGenerator() = default;

    KernelData get_kernel_data(const kernel_impl_params& params) const override;

protected:
    virtual Arguments get_arguments_desc(const kernel_impl_params& params) const;
    virtual JitConstants get_jit_constants(const kernel_impl_params& params) const;
    virtual std::string get_entry_point(const kernel_impl_params& params) const;
    virtual std::string get_build_options(const kernel_impl_params& params) const;

    std::string build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& entry_point) const;

    const std::string m_kernel_name;
    std::string m_stage_suffix;
};

}  // namespace ov::intel_gpu::cm
