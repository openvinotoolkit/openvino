// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"

namespace ov::intel_gpu::ocl {

class KernelGenerator : public KernelGeneratorBase {
public:
    KernelGenerator(const KernelGenerator&) = default;
    KernelGenerator(KernelGenerator&&) = delete;
    KernelGenerator& operator=(const KernelGenerator&) = delete;
    KernelGenerator& operator=(KernelGenerator&&) = delete;
    explicit KernelGenerator(std::string_view name, std::string_view suffix = "") : m_kernel_name(name), m_stage_suffix(suffix) {}
    virtual ~KernelGenerator() = default;

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override;

protected:
    [[nodiscard]] virtual Arguments get_arguments_desc(const RuntimeParams& params) const;
    [[nodiscard]] virtual JitConstants get_jit_constants(const RuntimeParams& params) const;
    [[nodiscard]] virtual std::string get_entry_point(const RuntimeParams& params) const;
    [[nodiscard]] virtual std::string get_build_options(const RuntimeParams& params) const;

    [[nodiscard]] JitConstants make_base_jit_constants(const RuntimeParams& params) const;
    [[nodiscard]] static JitConstants make_tensors_jit_constants(const RuntimeParams& params);
    [[nodiscard]] static std::string build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& entry_point);

private:
    std::string m_kernel_name;
    std::string m_stage_suffix;
};

}  // namespace ov::intel_gpu::ocl
