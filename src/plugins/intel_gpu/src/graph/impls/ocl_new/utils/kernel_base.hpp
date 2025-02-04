// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"

#include "primitive_db.h"
#include "jitter.hpp"
#include <string>

namespace ov::intel_gpu::ocl {

using namespace cldnn;

using primitive_db = kernel_selector::gpu::cache::primitive_db;

using KernelParams = cldnn::kernel_arguments_desc;

using KernelString = cldnn::kernel_string;
using WorkGroupSizes = cldnn::work_group_sizes;
using ScalarDescriptor = cldnn::scalar_desc;
using Scalars = cldnn::scalars_desc;
using ArgumentDescriptor = cldnn::argument_desc;
using Arguments = cldnn::arguments_desc;

struct KernelCode {
    std::shared_ptr<KernelString> kernelString;
};

struct DispatchData {
    WorkGroupSizes work_groups;
    Scalars scalars;
};

using InternalBuffers = std::vector<layout>;
using DispatchDataFunc = std::function<DispatchData(const kernel_impl_params&)>;
using GetInternalBuffersFunc = std::function<InternalBuffers(const kernel_impl_params&)>;
#define DISPATCH_DATA_FUNC(params, ...) [__VA_ARGS__](const kernel_impl_params& params) -> DispatchData
#define INTERNAL_BUFFERS_FUNC(params, ...) [__VA_ARGS__](const kernel_impl_params& params) -> InternalBuffers

struct KernelData {
    KernelCode code;
    KernelParams params;
    // std::vector<std::shared_ptr<micro::MicroKernelPackage>> micro_kernels;
    DispatchDataFunc update_dispatch_data_func = nullptr;

    std::vector<layout> internal_buffers;
    WeightsReorderParams weightsReorderParams;
};


using KernelsData = std::vector<KernelData>;

class KernelGeneratorBase {
public:
    KernelGeneratorBase() = default;
    virtual ~KernelGeneratorBase() = default;

    virtual KernelData get_kernel_data(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual GetInternalBuffersFunc get_interanl_buffers_func(const program_node& node, const kernel_impl_params& params) const { return nullptr; }
    virtual DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const = 0;
};


class SingleKernelGenerator : public KernelGeneratorBase {
public:
    explicit SingleKernelGenerator(const std::string name) : KernelGeneratorBase(), m_kernel_name(name) {}
    virtual ~SingleKernelGenerator() = default;

    KernelData get_kernel_data(const program_node& node, const kernel_impl_params& params) const override;
    const std::string get_name() const { return m_kernel_name; }

    void add_common_jit_constants(const JitConstants& jit_constants);
protected:

    virtual DispatchData get_dispatch_data(const kernel_impl_params& params) const;
    virtual Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const;
    virtual JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    virtual std::string get_entry_point(const program_node& node, const kernel_impl_params& params) const;
    virtual std::string get_build_options(const program_node& node, const kernel_impl_params& params) const;

    JitConstants make_base_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    std::string build_code(const std::string& template_name, const JitConstants& jit_constants, const std::string& entry_point) const;

    const std::string m_kernel_name;
    JitConstants m_jit_constants;
};

}  // namespace ov::intel_gpu::ocl
