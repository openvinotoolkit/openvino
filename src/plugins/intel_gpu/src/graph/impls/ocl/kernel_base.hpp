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

struct KernelCode {
    std::shared_ptr<KernelString> kernelString;
};

struct DispatchData {
    WorkGroupSizes work_groups;
    Scalars scalars;
};
using DispatchDataFunc = std::function<DispatchData(const kernel_impl_params&)>;
#define DISPATCH_DATA_FUNC(params, ...) [__VA_ARGS__](const kernel_impl_params& params) -> DispatchData

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

    virtual KernelsData get_kernels_data(const program_node& node, const kernel_impl_params& params) const = 0;

    static const primitive_db& get_db() { return db; }

protected:
    static const primitive_db db;
};


class SingleKernelGenerator : public KernelGeneratorBase {
public:
    explicit SingleKernelGenerator(const std::string name) : KernelGeneratorBase(), m_kernel_name(name) {}
    virtual ~SingleKernelGenerator() = default;

    KernelsData get_kernels_data(const program_node& node, const kernel_impl_params& params) const override {
        return { get_kernel_data(node, params) };
    }
    virtual KernelData get_kernel_data(const program_node& node, const kernel_impl_params& params) const;

    const std::string get_name() const { return m_kernel_name; }

    void add_common_jit_constants(const JitConstants& jit_constants);
protected:
    virtual DispatchDataFunc get_dispatch_data_func(const kernel_impl_params& params) const = 0;

    virtual DispatchData get_dispatch_data(const kernel_impl_params& params) const;
    virtual Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const;
    virtual JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    virtual std::string get_entry_point(const program_node& node, const kernel_impl_params& params) const;
    virtual std::string get_build_options(const program_node& node, const kernel_impl_params& params) const;
    virtual std::vector<layout> get_interanl_buffers(const program_node& node, const kernel_impl_params& params) const { return {}; }

    JitConstants make_base_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    std::string build_code(const std::string& template_name, const JitConstants& jit_constants, const std::string& entry_point) const;

    const std::string m_kernel_name;
    JitConstants m_jit_constants;
};

class MultiStageKernelGenerator : public KernelGeneratorBase {
public:
    template<typename... Stages>
    explicit MultiStageKernelGenerator(Stages... stages) : KernelGeneratorBase() {
        add_stages(std::forward<Stages>(stages)...);
    }

    virtual ~MultiStageKernelGenerator() = default;

    KernelsData get_kernels_data(const program_node& node, const kernel_impl_params& params) const override {
        KernelsData kds;
        for (auto& k : m_kernels) {
            k->add_common_jit_constants(get_jit_constants(node, params));
            kds.push_back(k->get_kernel_data(node, params));
        }

        return kds;
    }

    virtual JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const { return {}; }

protected:
    template<typename CurrentStage, typename... OtherStages>
    void add_stages(CurrentStage current, OtherStages... others) {
        add_stages(current);
        add_stages(std::forward<OtherStages>(others)...);
    }
    template<typename CurrentStage>
    void add_stages(CurrentStage current) {
        m_kernels.push_back(std::make_shared<CurrentStage>(current));
    }

    std::vector<std::shared_ptr<SingleKernelGenerator>> m_kernels;
};

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
