// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"

#include "primitive_db.h"
#include "jitter.hpp"
#include <cstddef>
#include <string>

namespace micro {
struct MicroKernelPackage;
}  // namspace

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
    std::shared_ptr<KernelString> kernel_string;
};

struct KernelData;
struct RuntimeParams { };

struct DispatchDataFunc {
    std::function<void(const kernel_impl_params&, KernelData&, RuntimeParams*)> m_dispatch_data_func = nullptr;

    template <typename Callable>
    DispatchDataFunc(Callable&& func) : m_dispatch_data_func(std::forward<Callable>(func)) {}
    explicit DispatchDataFunc(std::nullptr_t) {}

    void operator()(const kernel_impl_params& params, KernelData& kd, RuntimeParams* rt_params = nullptr) { m_dispatch_data_func(params, kd, rt_params); }
};

#define DISPATCH_DATA_FUNC(params, kd, rt_params, ...) [__VA_ARGS__](const kernel_impl_params& params, KernelData& kd, RuntimeParams* rt_params)

struct KernelData {
    KernelCode code;
    KernelParams params;
    std::vector<std::shared_ptr<micro::MicroKernelPackage>> micro_kernels;
    DispatchDataFunc update_dispatch_data_func{nullptr};
    WeightsReorderParams weights_reorder_params;
    bool need_args_update{true};

    void save(cldnn::BinaryOutputBuffer& ob) const;
    void load(cldnn::BinaryInputBuffer& ib);
};

using KernelsData = std::vector<KernelData>;

class KernelGeneratorBase {
public:
    KernelGeneratorBase() = default;
    virtual ~KernelGeneratorBase() = default;

    virtual KernelData get_kernel_data(const kernel_impl_params& params) const = 0;
    virtual DispatchDataFunc get_dispatch_data_func() const = 0;
};

class SingleKernelGenerator : public KernelGeneratorBase {
public:
    explicit SingleKernelGenerator(std::string_view name) : KernelGeneratorBase(), m_kernel_name(name) {}
    virtual ~SingleKernelGenerator() = default;

    KernelData get_kernel_data(const kernel_impl_params& params) const override;

protected:
    virtual Arguments get_arguments_desc(const kernel_impl_params& params) const;
    virtual JitConstants get_jit_constants(const kernel_impl_params& params) const;
    virtual std::string get_entry_point(const kernel_impl_params& params) const;
    virtual std::string get_build_options(const kernel_impl_params& params) const;

    JitConstants make_base_jit_constants(const kernel_impl_params& params) const;
    JitConstants make_tensors_jit_constants(const kernel_impl_params& params) const ;
    std::string build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& entry_point) const;

    const std::string m_kernel_name;
    std::string m_stage_suffix;
};

}  // namespace ov::intel_gpu::ocl
