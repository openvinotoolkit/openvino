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

struct KernelData {
    KernelCode code;
    KernelParams params;
    // std::vector<std::shared_ptr<micro::MicroKernelPackage>> micro_kernels;
    std::function<void(const program_node&, const kernel_impl_params&)> update_dispatch_data_func = nullptr;

    std::vector<size_t> internal_buffer_sizes;
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

protected:
    virtual WorkGroupSizes get_dispatch_data(const program_node& node, const kernel_impl_params& params) const = 0;

    virtual Arguments get_arguments_desc(const program_node& node, const kernel_impl_params& params) const;
    virtual JitConstants get_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    virtual std::string get_entry_point(const program_node& node, const kernel_impl_params& params) const;

    JitConstants make_base_jit_constants(const program_node& node, const kernel_impl_params& params) const;
    std::string build_code(const std::string& template_name, const JitConstants& jit_constants, const std::string& entry_point) const;

    const std::string m_kernel_name;
};

class MultiKernelGenerator : public KernelGeneratorBase {
public:
    template<typename... Stages>
    explicit MultiKernelGenerator(Stages... stages) : KernelGeneratorBase() {
        add_stages(std::forward<Stages>(stages)...);
    }

    virtual ~MultiKernelGenerator() = default;

    KernelsData get_kernels_data(const program_node& node, const kernel_impl_params& params) const override {
        KernelsData kds;
        for (auto& k : m_kernels)
            kds.push_back(k->get_kernel_data(node, params));

        return kds;
    }

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
