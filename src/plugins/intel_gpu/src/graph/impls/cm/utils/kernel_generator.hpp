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
    // kernel name is the id used to find kernel in database while suffix is an optional identifier of specific stage
    // for multi-stage implementations which makes kernel names more clear in any kind of profiles
    explicit KernelGenerator(std::string_view name, std::string_view suffix = "") : m_kernel_name(name), m_stage_suffix(suffix) {}
    virtual ~KernelGenerator() = default;

    // Code generator is not supposed to be copied/moved as it's mainly used once to produce KernelData
    // or to query DispatchDataFunc during importing compiled blob. After that generators can be removed
    KernelGenerator(const KernelGenerator&) = delete;
    KernelGenerator(KernelGenerator&&) = delete;
    KernelGenerator& operator=(const KernelGenerator&) = delete;
    KernelGenerator& operator=(KernelGenerator&&) = delete;

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override;

protected:
    // Defines mapping between kernel argument and primitive_inst's memory buffers
    // Count of elements in the vector must match count of kernel arguments
    [[nodiscard]] virtual Arguments get_arguments_desc(const RuntimeParams& params) const;

    // Extra jit constants that are added to kernel template to specialize it for the particular use case
    [[nodiscard]] virtual JitConstants get_jit_constants(const RuntimeParams& params) const;

    // Returns actual name of the kernel
    [[nodiscard]] virtual std::string get_entry_point(const RuntimeParams& params) const;

    // Returns build options that are passed to CM compiler
    [[nodiscard]] virtual std::string get_build_options(const RuntimeParams& params) const;

    [[nodiscard]] static std::string build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& entry_point);

private:
    std::string m_kernel_name;
    std::string m_stage_suffix;
};

}  // namespace ov::intel_gpu::cm
