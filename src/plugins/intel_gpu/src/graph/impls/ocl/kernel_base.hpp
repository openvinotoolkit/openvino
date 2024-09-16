// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"

#include "primitive_db.h"
#include "jitter.hpp"
#include <string>

namespace ov {
namespace intel_gpu {
namespace ocl {

using namespace cldnn;

using primitive_db = kernel_selector::gpu::cache::primitive_db;

using KernelParams = cldnn::kernel_arguments_desc;

using KernelString = cldnn::kernel_string;
using WorkGroupSizes = cldnn::work_group_sizes;
using ScalarDescriptor = cldnn::scalar_desc;
using Scalars = cldnn::scalars_desc;
using ArgumentDescriptor = cldnn::argument_desc;
using Arguments = cldnn::arguments_desc;
using KernelParams = cldnn::kernel_arguments_desc;

struct KernelCode {
    std::shared_ptr<KernelString> kernelString;
};

struct KernelData {
    KernelCode code;
    KernelParams params;
    // std::vector<std::shared_ptr<micro::MicroKernelPackage>> micro_kernels;
    std::function<void(const program_node&, const kernel_impl_params&)> update_dispatch_data_func = nullptr;

    std::vector<size_t> internal_buffer_sizes;
    WeightsReorderParams weightsReorderParams;
};

class KernelGeneratorBase {
public:
    explicit KernelGeneratorBase(const std::string name) : m_kernel_name(name) {}
    virtual ~KernelGeneratorBase() = default;

    virtual KernelData get_kernel_data(const program_node& node, const kernel_impl_params& params) const = 0;

    virtual const std::string get_name() const { return m_kernel_name; }
    static const primitive_db& get_db() { return db; }

protected:
    virtual JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual std::string get_entry_point(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const = 0;

    JitConstants make_base_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    std::string build_code(const std::string& template_name, const JitConstants& jit_constants, const std::string& entry_point) const;

    static const primitive_db db;
    const std::string m_kernel_name;
};

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
